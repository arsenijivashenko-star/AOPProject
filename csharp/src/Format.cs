using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace Sarc;

public sealed class VolHeader
{
    public int Idx, Tot;
    public byte[] Salt = Array.Empty<byte>();
    public byte[] Nonce = Array.Empty<byte>();
    public int N;                       // ciphertext length
}

public sealed class FileEntry
{
    public string Name = "";
    public uint Size, Crc;
    public byte Type, Strat;
}

public sealed class BlockEntry
{
    public int Fi;                      // index into this volume's file table
    public uint Off;                    // block index inside the file (0,1,2,...)
    public uint Raw, Coded;
    public byte Flag;                   // 0 = Huffman/LZ, 1 = stored (NO_COMPRESSION)
    public byte[] Data = Array.Empty<byte>();
}

/// <summary>Growable byte buffer used as the decoded file history.</summary>
public sealed class ByteBuf
{
    public byte[] A = new byte[1024];
    public int N;
    public void Add(byte b)
    {
        if (N == A.Length)
        {
            if (A.Length >= 0x40000000) throw new SarcException("E_PAYLOAD", "file too large");
            Array.Resize(ref A, A.Length * 2);
        }
        A[N++] = b;
    }
    public void AddRange(byte[] s)
    {
        foreach (byte b in s) Add(b);
    }
    public byte[] ToArray() => A.AsSpan(0, N).ToArray();
}

public static class Fmt
{
    public const int HDR = 41, TAG = 32;

    // ---------------- crypto ----------------
    /// <summary>PBKDF2-HMAC-SHA256, 100 000 iterations, 64 bytes: K_enc = first 32, K_mac = last 32.</summary>
    public static (byte[] ke, byte[] km) Derive(byte[] password, byte[] salt)
    {
        byte[] k = Rfc2898DeriveBytes.Pbkdf2(password, salt, 100000, HashAlgorithmName.SHA256, 64);
        return (k[..32], k[32..]);
    }

    /// <summary>AES-256-CTR: counter block = nonce(12) || uint32_BE(i), i starts at 1.</summary>
    public static byte[] Ctr(byte[] key, byte[] nonce, byte[] data)
    {
        var res = new byte[data.Length];
        using var aes = Aes.Create();
        aes.Key = key;
        var ctr = new byte[16];
        Buffer.BlockCopy(nonce, 0, ctr, 0, 12);
        var ks = new byte[16];
        uint i = 1;
        for (int p = 0; p < data.Length; p += 16)
        {
            BinaryPrimitives.WriteUInt32BigEndian(ctr.AsSpan(12), i++);
            aes.EncryptEcb(ctr, ks, PaddingMode.None);
            int m = Math.Min(16, data.Length - p);
            for (int j = 0; j < m; j++) res[p + j] = (byte)(data[p + j] ^ ks[j]);
        }
        return res;
    }

    public static bool SafeName(string n)
    {
        if (string.IsNullOrEmpty(n)) return false;
        if (n.Contains('/') || n.Contains('\\') || n.Contains(':') || n.Contains('\0')) return false;
        if (n == "." || n == "..") return false;
        return true;
    }

    // ---------------- volume ----------------
    /// <summary>Validates the public 41-byte header (magic, version, lengths) BEFORE any decryption.</summary>
    public static VolHeader ParseHeader(byte[] d)
    {
        if (d.Length < HDR + TAG) throw new SarcException("E_LENGTH", "too short");
        if (d[0] != 'S' || d[1] != 'A' || d[2] != 'R' || d[3] != 'C') throw new SarcException("E_MAGIC");
        if (d[4] != 1) throw new SarcException("E_VERSION");
        int idx = BinaryPrimitives.ReadUInt16BigEndian(d.AsSpan(5));
        int tot = BinaryPrimitives.ReadUInt16BigEndian(d.AsSpan(7));
        uint n = BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(37));
        if ((long)HDR + n + TAG != d.Length) throw new SarcException("E_LENGTH", $"{(long)HDR + n + TAG} != {d.Length}");
        if (tot == 0 || idx >= tot) throw new SarcException("E_PAYLOAD", "volume index");
        return new VolHeader { Idx = idx, Tot = tot, Salt = d[9..25], Nonce = d[25..37], N = (int)n };
    }

    /// <summary>Header check -> HMAC (over header+ciphertext) -> decrypt. keyCache lets all volumes of one archive share one PBKDF2 run.</summary>
    public static (VolHeader h, byte[] payload) OpenVolume(byte[] d, string password,
        Dictionary<string, (byte[] ke, byte[] km)>? keyCache = null)
    {
        VolHeader h = ParseHeader(d);
        string ck = Convert.ToHexString(h.Salt);
        (byte[] ke, byte[] km) keys;
        if (keyCache == null || !keyCache.TryGetValue(ck, out keys))
        {
            keys = Derive(Encoding.UTF8.GetBytes(password), h.Salt);
            if (keyCache != null) keyCache[ck] = keys;
        }
        byte[] tag = HMACSHA256.HashData(keys.km, d.AsSpan(0, HDR + h.N));
        if (!CryptographicOperations.FixedTimeEquals(tag, d.AsSpan(HDR + h.N, TAG))) throw new SarcException("E_HMAC");
        byte[] ct = d.AsSpan(HDR, h.N).ToArray();
        return (h, Ctr(keys.ke, h.Nonce, ct));
    }

    // ---------------- payload ----------------
    private sealed class PReader
    {
        private readonly byte[] _b;
        public int Pos;
        public PReader(byte[] b, int pos = 0) { _b = b; Pos = pos; }
        public int Length => _b.Length;
        private void Need(long n)
        {
            if (n < 0 || Pos + n > _b.Length) throw new SarcException("E_PAYLOAD", "truncated");
        }
        public byte U8() { Need(1); return _b[Pos++]; }
        public ushort U16() { Need(2); var v = BinaryPrimitives.ReadUInt16BigEndian(_b.AsSpan(Pos)); Pos += 2; return v; }
        public uint U32() { Need(4); var v = BinaryPrimitives.ReadUInt32BigEndian(_b.AsSpan(Pos)); Pos += 4; return v; }
        public byte[] Bytes(long n) { Need(n); var r = _b.AsSpan(Pos, (int)n).ToArray(); Pos += (int)n; return r; }
    }

    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    /// <summary>Layout: nfiles u16 | reserved u16 (=0) | files{name_len u16, name, size u32, crc u32, type u8, strategy u8}
    /// | nblocks u16 | blocks{file_idx u16, block_idx u32, raw u32, coded u32, flag u8} | block data. Big-endian.</summary>
    public static (List<FileEntry> files, List<BlockEntry> blocks) ParsePayload(byte[] pt)
    {
        var r = new PReader(pt);
        int nfiles = r.U16();
        if (r.U16() != 0) throw new SarcException("E_PAYLOAD", "reserved != 0");
        var files = new List<FileEntry>();
        for (int i = 0; i < nfiles; i++)
        {
            int nl = r.U16();
            byte[] raw = r.Bytes(nl);
            string name;
            try { name = StrictUtf8.GetString(raw); }
            catch (ArgumentException) { throw new SarcException("E_PAYLOAD", "name encoding"); }
            var f = new FileEntry { Name = name, Size = r.U32(), Crc = r.U32(), Type = r.U8(), Strat = r.U8() };
            if (!SafeName(name)) throw new SarcException("E_PAYLOAD", "unsafe name");
            files.Add(f);
        }
        int nblocks = r.U16();
        var blocks = new List<BlockEntry>();
        for (int i = 0; i < nblocks; i++)
        {
            var b = new BlockEntry { Fi = r.U16(), Off = r.U32(), Raw = r.U32(), Coded = r.U32(), Flag = r.U8() };
            if (b.Fi >= nfiles) throw new SarcException("E_PAYLOAD", "file idx");
            blocks.Add(b);
        }
        foreach (var b in blocks) b.Data = r.Bytes(b.Coded);
        if (r.Pos != r.Length) throw new SarcException("E_PAYLOAD", "trailing bytes");
        return (files, blocks);
    }

    // ---------------- canonical Huffman ----------------
    /// <summary>Canonical codes: symbols sorted by (length, symbol); only the LENGTHS are stored in the archive.</summary>
    public static Dictionary<int, (ulong code, int len)> CanonCodes(List<(int sym, int len)> items)
    {
        var sorted = items.OrderBy(t => t.len).ThenBy(t => t.sym).ToList();
        var codes = new Dictionary<int, (ulong, int)>();
        ulong code = 0; int prev = 0;
        foreach (var (s, l) in sorted)
        {
            code <<= (l - prev); prev = l;
            codes[s] = (code, l);
            code++;
        }
        return codes;
    }

    private static ulong Key(int len, ulong code) => ((ulong)len << 40) | code;

    private static List<(int sym, int len)> ReadTree(PReader r)
    {
        int cnt = r.U16();
        var ent = new List<(int, int)>(cnt);
        for (int i = 0; i < cnt; i++) { int s = r.U16(); int l = r.U8(); ent.Add((s, l)); }
        return ent;
    }

    private static Dictionary<ulong, int> DecoderTable(List<(int sym, int len)> ent)
    {
        var tab = new Dictionary<ulong, int>();
        if (ent.Count == 0) return tab;
        if (ent.Count == 1) { tab[Key(1, 0)] = ent[0].sym; return tab; }      // degenerate alphabet: 1-bit code
        foreach (var (s, l) in ent) if (l < 1 || l > 32) throw new SarcException("E_PAYLOAD", "code length");
        foreach (var kv in CanonCodes(ent)) tab[Key(kv.Value.len, kv.Value.code)] = kv.Key;
        return tab;
    }

    private sealed class BitReader
    {
        private readonly byte[] _d; private readonly int _off; private readonly long _n; private long _p;
        public BitReader(byte[] d, int off, int len) { _d = d; _off = off; _n = (long)len * 8; }
        public int Bit()
        {
            if (_p >= _n) throw new SarcException("E_PAYLOAD", "bitstream end");
            int v = (_d[_off + (int)(_p >> 3)] >> (7 - (int)(_p & 7))) & 1; _p++; return v;
        }
    }

    private static int DecSym(BitReader br, Dictionary<ulong, int> tab)
    {
        ulong c = 0;
        for (int l = 1; l <= 32; l++)
        {
            c = (c << 1) | (uint)br.Bit();
            if (tab.TryGetValue(Key(l, c), out int s)) return s;
        }
        throw new SarcException("E_PAYLOAD", "bad code");
    }

    /// <summary>Decodes one Huffman+LZ block, appending to the file history buffer.
    /// Block: 3 trees (literal+marker256 / length-3 / distance slot), u32 tokens, u32 bytes, bitstream.</summary>
    public static void DecodeBlock(byte[] data, long rawLen, ByteBuf output)
    {
        var r = new PReader(data);
        var lt = ReadTree(r); var mt = ReadTree(r); var dt = ReadTree(r);
        uint ntok = r.U32(); uint nbytes = r.U32();
        if (nbytes > data.Length - r.Pos) throw new SarcException("E_PAYLOAD", "bitstream length");
        var br = new BitReader(data, r.Pos, (int)nbytes);
        var L = DecoderTable(lt); var M = DecoderTable(mt); var D = DecoderTable(dt);
        int start = output.N;
        for (uint t = 0; t < ntok; t++)
        {
            int s = DecSym(br, L);
            if (s < 256) output.Add((byte)s);
            else
            {
                int ln = DecSym(br, M) + 3;
                int ds = DecSym(br, D);
                long dist;
                if (ds == 0) dist = 1;
                else
                {
                    if (ds > 30) throw new SarcException("E_PAYLOAD", "distance slot");
                    long v = 1;
                    for (int k = 0; k < ds - 1; k++) v = (v << 1) | (uint)br.Bit();
                    dist = v + 1;
                }
                if (dist > output.N) throw new SarcException("E_PAYLOAD", "dist");
                for (int k = 0; k < ln; k++) { byte b = output.A[output.N - (int)dist]; output.Add(b); }
            }
            if (output.N - start > rawLen) throw new SarcException("E_PAYLOAD", "raw_len overflow");
        }
        if (output.N - start != rawLen) throw new SarcException("E_PAYLOAD", "raw_len mismatch");
    }
}
