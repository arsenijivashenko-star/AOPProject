using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace Sarc;

public sealed class SarcConfig
{
    public int MaxVolume = 65536;
    public CodecParams Params = new();
    public string Packer = "dp";                                   // "dp" (exact) or "greedy"
    public Func<byte[], int>? Classifier;                          // perceptron: first bytes of a file -> type 0..6
    public Dictionary<int, int> StrategyByType = new() { [5] = Strategy.Store };   // already-compressed data: do not compress
}

public sealed class Candidate
{
    public int File, Idx, Raw; public byte Flag; public byte[] Coded = Array.Empty<byte>(); public long Value;
}

public sealed class VolumeInfo { public int W, Table, Weight, N; public long Value; }

public sealed class ArchiveStats
{
    public int Volumes, Blocks; public long ArcBytes, OrigBytes, CodedBytes; public double Seconds;
    public List<FileEntry> Files = new(); public List<VolumeInfo> VolInfo = new();
}

public static class Archiver
{
    public static readonly int[] TypePriority = { 3, 3, 2, 2, 2, 1, 1 };   // team_config.json priorities by type
    public static int MaxNameBytes = 255;

    // ------------------------------------------------------------ create
    private static void ValidateInput(string path)
    {
        string name = Path.GetFileName(path);
        if (!File.Exists(path)) throw new SarcException("E_INPUT", $"not a file: {name}");
        try { using var _ = File.OpenRead(path); }
        catch (UnauthorizedAccessException) { throw new SarcException("E_INPUT", $"no read permission: {name}"); }
        catch (IOException e) { throw new SarcException("E_INPUT", $"cannot read {name}: {e.Message}"); }
        if (Encoding.UTF8.GetByteCount(name) > MaxNameBytes) throw new SarcException("E_INPUT", "file name too long");
        if (!Fmt.SafeName(name)) throw new SarcException("E_INPUT", $"unsafe file name: {name}");
    }

    public static List<string> Collect(string input, bool recursive)
    {
        if (File.Exists(input)) return new List<string> { input };
        if (!Directory.Exists(input)) throw new SarcException("E_INPUT", "input does not exist");
        var opt = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var paths = Directory.GetFiles(input, "*", opt).ToList();
        paths.Sort(StringComparer.Ordinal);
        var dup = paths.GroupBy(Path.GetFileName).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
        if (dup.Count > 0) throw new SarcException("E_INPUT", "duplicate file names (archive stores names only): " + string.Join(", ", dup.Take(3)));
        return paths;
    }

    private static int ReadFull(Stream s, byte[] buf)
    {
        int n = 0;
        while (n < buf.Length) { int r = s.Read(buf, n, buf.Length - n); if (r <= 0) break; n += r; }
        return n;
    }

    /// <summary>Analysis (perceptron) -> parse -> Huffman, block by block. All blocks are encoded BEFORE packing, so weights are fixed.</summary>
    public static (List<FileEntry> files, List<Candidate> cands) AnalyzeAndEncode(List<string> paths, SarcConfig cfg)
    {
        var files = new List<FileEntry>(); var cands = new List<Candidate>(); int bs = cfg.Params.BlockSize;
        if (bs < 16 || bs > 65536) throw new SarcException("E_CAPACITY", "block size must be in 16..65536");
        for (int fi = 0; fi < paths.Count; fi++)
        {
            ValidateInput(paths[fi]);
            using var f = File.OpenRead(paths[fi]);
            long size = f.Length;
            if (size > int.MaxValue) throw new SarcException("E_INPUT", "file larger than 2 GiB is not supported");
            var head = new byte[Math.Min(size, 65536)];
            ReadFull(f, head);
            int ftype = cfg.Classifier != null ? cfg.Classifier(head) : 6;
            if (ftype < 0 || ftype > 6) ftype = 6;
            int strat = cfg.StrategyByType.TryGetValue(ftype, out int st) ? st : cfg.Params.Strategy;
            var p = cfg.Params.Clone(); p.Strategy = strat;
            f.Seek(0, SeekOrigin.Begin);
            uint crc = 0; long total = 0; int bi = 0;
            var buf = new byte[bs];
            while (true)
            {
                int n = ReadFull(f, buf);
                if (n == 0 && bi > 0) break;
                byte[] raw = buf.AsSpan(0, n).ToArray();
                crc = Crc32.Compute(raw, crc); total += n;
                var (flag, coded) = Codec.EncodeBlock(raw, p);
                cands.Add(new Candidate { File = fi, Idx = bi, Raw = n, Flag = flag, Coded = coded, Value = (long)TypePriority[ftype] * (n + 1) });
                bi++;
                if (n == 0) break;
            }
            files.Add(new FileEntry { Name = Path.GetFileName(paths[fi]), Size = (uint)total, Crc = crc, Type = (byte)ftype, Strat = (byte)strat });
        }
        return (files, cands);
    }

    private static int TableBytes(FileEntry f) => 2 + Encoding.UTF8.GetByteCount(f.Name) + 10;

    /// <summary>Before every volume: fix the set of files that still have unplaced blocks, reserve their whole table, then
    /// W = max_volume - 41 - 32 - 6 - table. Solve a 0/1 knapsack (weight = coded_len + 15, value = priority * raw_len).</summary>
    public static (List<(List<int> fset, List<Candidate> blocks)> plan, List<VolumeInfo> info) PlanVolumes(
        List<FileEntry> files, List<Candidate> cands, int maxVolume, string packer)
    {
        var remaining = new List<Candidate>(cands);
        var plan = new List<(List<int>, List<Candidate>)>(); var info = new List<VolumeInfo>();
        while (remaining.Count > 0)
        {
            var fset = remaining.Select(c => c.File).Distinct().OrderBy(x => x).ToList();
            int table = fset.Sum(i => TableBytes(files[i]));
            int W = maxVolume - Fmt.HDR - Fmt.TAG - 6 - table;
            if (W <= 0 || fset.Count > 65535)
                throw new SarcException("E_CAPACITY", $"volume size {maxVolume} too small for the file table ({table} bytes)");
            var w = remaining.Select(c => c.Coded.Length + 15).ToList();
            var v = remaining.Select(c => c.Value).ToList();
            var pick = packer == "dp" ? Knapsack.Dp(v, w, W) : Knapsack.Greedy(v, w, W);
            if (pick.Count == 0)
                throw new SarcException("E_CAPACITY", $"no block fits: smallest weight {w.Min()} > W={W}; reduce block size or raise max volume");
            if (pick.Count > 65535) throw new SarcException("E_CAPACITY", "too many blocks in a volume");
            info.Add(new VolumeInfo { W = W, Table = table, Value = Knapsack.Total(v, pick), Weight = pick.Sum(i => w[i]), N = pick.Count });
            var ps = new HashSet<int>(pick);
            plan.Add((fset, pick.Select(i => remaining[i]).ToList()));
            remaining = remaining.Where((c, i) => !ps.Contains(i)).ToList();
        }
        return (plan, info);
    }

    private static void W16(List<byte> o, int v) { o.Add((byte)(v >> 8)); o.Add((byte)v); }
    private static void W32(List<byte> o, uint v) { o.Add((byte)(v >> 24)); o.Add((byte)(v >> 16)); o.Add((byte)(v >> 8)); o.Add((byte)v); }

    public static byte[] BuildPayload(List<FileEntry> files, List<int> fset, List<Candidate> blocks)
    {
        var pos = new Dictionary<int, int>();
        for (int k = 0; k < fset.Count; k++) pos[fset[k]] = k;
        var o = new List<byte>();
        W16(o, fset.Count); W16(o, 0);
        foreach (int fi in fset)
        {
            var f = files[fi]; byte[] nb = Encoding.UTF8.GetBytes(f.Name);
            W16(o, nb.Length); o.AddRange(nb); W32(o, f.Size); W32(o, f.Crc); o.Add(f.Type); o.Add(f.Strat);
        }
        var sorted = blocks.OrderBy(c => c.File).ThenBy(c => c.Idx).ToList();
        W16(o, sorted.Count);
        foreach (var c in sorted) { W16(o, pos[c.File]); W32(o, (uint)c.Idx); W32(o, (uint)c.Raw); W32(o, (uint)c.Coded.Length); o.Add(c.Flag); }
        foreach (var c in sorted) o.AddRange(c.Coded);
        return o.ToArray();
    }

    /// <summary>Encrypt-then-MAC: header(41) || AES-CTR ciphertext || HMAC-SHA256(header || ciphertext).</summary>
    public static byte[] SealVolume(byte[] payload, byte[] ke, byte[] km, byte[] salt, byte[] noncePrefix, int idx, int total)
    {
        var nonce = new byte[12];
        Buffer.BlockCopy(noncePrefix, 0, nonce, 0, 8);
        BinaryPrimitives.WriteUInt32BigEndian(nonce.AsSpan(8), (uint)idx);
        byte[] ct = Fmt.Ctr(ke, nonce, payload);
        var vol = new byte[Fmt.HDR + ct.Length + Fmt.TAG];
        vol[0] = (byte)'S'; vol[1] = (byte)'A'; vol[2] = (byte)'R'; vol[3] = (byte)'C'; vol[4] = 1;
        BinaryPrimitives.WriteUInt16BigEndian(vol.AsSpan(5), (ushort)idx);
        BinaryPrimitives.WriteUInt16BigEndian(vol.AsSpan(7), (ushort)total);
        Buffer.BlockCopy(salt, 0, vol, 9, 16);
        Buffer.BlockCopy(nonce, 0, vol, 25, 12);
        BinaryPrimitives.WriteUInt32BigEndian(vol.AsSpan(37), (uint)ct.Length);
        Buffer.BlockCopy(ct, 0, vol, Fmt.HDR, ct.Length);
        byte[] tag = HMACSHA256.HashData(km, vol.AsSpan(0, Fmt.HDR + ct.Length));
        Buffer.BlockCopy(tag, 0, vol, Fmt.HDR + ct.Length, Fmt.TAG);
        return vol;
    }

    /// <summary>Creates the archive; returns the written volume paths. One volume: base.arc; several: base.part0.arc, base.part1.arc, ...</summary>
    public static List<string> Create(string input, string outBase, string password, SarcConfig? cfg = null,
        bool recursive = false, ArchiveStats? stats = null)
    {
        cfg ??= new SarcConfig();
        if (string.IsNullOrEmpty(password)) throw new SarcException("E_INPUT", "empty password");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var paths = Collect(input, recursive);
        if (paths.Count == 0) throw new SarcException("E_INPUT", "no input files");
        if (paths.Count > 65535) throw new SarcException("E_INPUT", "too many files");
        var (files, cands) = AnalyzeAndEncode(paths, cfg);
        var (plan, info) = PlanVolumes(files, cands, cfg.MaxVolume, cfg.Packer);
        if (plan.Count > 65535) throw new SarcException("E_CAPACITY", "too many volumes");
        byte[] salt = RandomNumberGenerator.GetBytes(16), noncePrefix = RandomNumberGenerator.GetBytes(8);   // fresh per archive
        var (ke, km) = Fmt.Derive(Encoding.UTF8.GetBytes(password), salt);                                      // PBKDF2 once per archive
        var vols = new List<byte[]>();
        for (int i = 0; i < plan.Count; i++)
            vols.Add(SealVolume(BuildPayload(files, plan[i].fset, plan[i].blocks), ke, km, salt, noncePrefix, i, plan.Count));
        string dir = Path.GetDirectoryName(Path.GetFullPath(outBase))!;
        Directory.CreateDirectory(dir);
        var outs = new List<string>();
        for (int i = 0; i < vols.Count; i++)
        {
            string p = vols.Count == 1 ? outBase + ".arc" : $"{outBase}.part{i}.arc";
            string tmp = p + ".tmp";
            File.WriteAllBytes(tmp, vols[i]);
            File.Move(tmp, p, true);
            outs.Add(p);
        }
        if (stats != null)
        {
            stats.Volumes = vols.Count; stats.VolInfo = info; stats.ArcBytes = vols.Sum(v => (long)v.Length);
            stats.OrigBytes = files.Sum(f => (long)f.Size); stats.Seconds = sw.Elapsed.TotalSeconds; stats.Files = files;
            stats.Blocks = cands.Count; stats.CodedBytes = cands.Sum(c => (long)c.Coded.Length);
        }
        return outs;
    }

    // ------------------------------------------------------------ extract
    public static List<string> FindVolumes(IReadOnlyList<string> args)
    {
        if (args.Count > 1) return args.ToList();
        string a = args[0];
        if (File.Exists(a)) return new List<string> { a };
        string dir = Path.GetDirectoryName(Path.GetFullPath(a))!, stem = Path.GetFileName(a);
        var parts = Directory.Exists(dir)
            ? Directory.GetFiles(dir, stem + ".part*.arc").Where(p => int.TryParse(
                  Path.GetFileName(p).Substring(stem.Length + 5, Path.GetFileName(p).Length - stem.Length - 9), out _)).ToList()
            : new List<string>();
        if (parts.Count > 0)
        {
            parts.Sort((x, y) => PartNo(x, stem).CompareTo(PartNo(y, stem)));
            return parts;
        }
        if (File.Exists(a + ".arc")) return new List<string> { a + ".arc" };
        throw new SarcException("E_INPUT", "archive not found");
    }

    private static int PartNo(string path, string stem)
    {
        string n = Path.GetFileName(path);
        return int.Parse(n.Substring(stem.Length + 5, n.Length - stem.Length - 9));
    }

    private sealed class Meta { public uint Size, Crc; public byte Type; }

    /// <summary>Decodes and VERIFIES everything in memory (header -> HMAC -> payload -> CRC32); writes nothing.</summary>
    public static Dictionary<string, byte[]> DecodeArchive(IReadOnlyList<string> volPaths, string password)
        => DecodeVolumes(volPaths.Select(File.ReadAllBytes).ToList(), password);

    public static Dictionary<string, byte[]> DecodeVolumes(List<byte[]> volumes, string password)
    {
        if (volumes.Count == 0) throw new SarcException("E_INPUT", "no volumes");
        var cache = new Dictionary<string, (byte[] ke, byte[] km)>();
        var seen = new HashSet<int>(); int tot = -1;
        var metas = new Dictionary<string, Meta>(); var blocks = new Dictionary<string, SortedDictionary<uint, BlockEntry>>();
        foreach (byte[] d in volumes)
        {
            var (h, pt) = Fmt.OpenVolume(d, password, cache);
            if (tot < 0) tot = h.Tot;
            else if (h.Tot != tot) throw new SarcException("E_PAYLOAD", "volumes belong to different sets");
            if (!seen.Add(h.Idx)) throw new SarcException("E_VOLUME_DUP", $"volume {h.Idx} supplied twice");
            var (fl, bl) = Fmt.ParsePayload(pt);
            foreach (var f in fl)
            {
                if (metas.TryGetValue(f.Name, out var m))
                {
                    if (m.Size != f.Size || m.Crc != f.Crc) throw new SarcException("E_PAYLOAD", "inconsistent file table");
                }
                else metas[f.Name] = new Meta { Size = f.Size, Crc = f.Crc, Type = f.Type };
            }
            foreach (var b in bl)
            {
                string name = fl[b.Fi].Name;
                if (!blocks.TryGetValue(name, out var bd)) blocks[name] = bd = new SortedDictionary<uint, BlockEntry>();
                if (!bd.TryAdd(b.Off, b)) throw new SarcException("E_PAYLOAD", "duplicate block");
            }
        }
        var missing = Enumerable.Range(0, tot).Where(i => !seen.Contains(i)).ToList();
        if (missing.Count > 0) throw new SarcException("E_VOLUME_MISSING", "missing volume(s) " + string.Join(",", missing));
        var result = new Dictionary<string, byte[]>();
        foreach (var (name, m) in metas)
        {
            blocks.TryGetValue(name, out var bl);
            uint expect = 0; var buf = new ByteBuf();
            if (bl != null)
                foreach (var (idx, b) in bl)
                {
                    if (idx != expect++) throw new SarcException("E_PAYLOAD", "block gap");
                    if (b.Flag == 1)
                    {
                        if (b.Raw != b.Coded) throw new SarcException("E_PAYLOAD", "stored block length");
                        buf.AddRange(b.Data);
                    }
                    else if (b.Flag == 0) Fmt.DecodeBlock(b.Data, b.Raw, buf);
                    else throw new SarcException("E_PAYLOAD", "block flag");
                    if (buf.N > m.Size) throw new SarcException("E_PAYLOAD", "size overflow");
                }
            if (buf.N != m.Size) throw new SarcException("E_PAYLOAD", "size mismatch");
            if (Crc32.Compute(buf.A.AsSpan(0, buf.N)) != m.Crc) throw new SarcException("E_CRC", name);
            result[name] = buf.ToArray();
        }
        return result;
    }

    /// <summary>Atomic extraction: nothing is written unless EVERY volume and EVERY file verified; files are staged in a temp
    /// directory and then moved into place.</summary>
    public static List<string> Extract(IReadOnlyList<string> archive, string outDir, string password)
    {
        var data = DecodeArchive(FindVolumes(archive), password);
        Directory.CreateDirectory(outDir);
        string tmp = Path.Combine(outDir, ".sarc_" + Convert.ToHexString(RandomNumberGenerator.GetBytes(6)));
        Directory.CreateDirectory(tmp);
        try
        {
            foreach (var (n, b) in data)
            {
                if (!Fmt.SafeName(n)) throw new SarcException("E_PAYLOAD", "unsafe name");
                File.WriteAllBytes(Path.Combine(tmp, n), b);
            }
            foreach (var n in data.Keys) File.Move(Path.Combine(tmp, n), Path.Combine(outDir, n), true);
        }
        finally { try { Directory.Delete(tmp, true); } catch (IOException) { } }
        return data.Keys.OrderBy(x => x, StringComparer.Ordinal).ToList();
    }
}
