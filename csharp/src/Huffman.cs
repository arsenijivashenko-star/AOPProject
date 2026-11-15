using System.Buffers.Binary;

namespace Sarc;

/// <summary>LZ77 token: Len == 0 -> literal Lit; otherwise reference (Len, Dist).</summary>
public readonly record struct Tok(int Len, int Dist, byte Lit);

/// <summary>Huffman coding from scratch: priority-queue tree -> code lengths -> canonical codes (only lengths are stored).</summary>
public static class Huffman
{
    public static int BitLen(int x) => 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)x);

    /// <summary>Optimal prefix-code lengths by the greedy "merge two lightest" rule. Single symbol -> length 1.</summary>
    public static int[] CodeLengths(long[] freq)
    {
        var lens = new int[freq.Length];
        var syms = new List<int>();
        for (int s = 0; s < freq.Length; s++) if (freq[s] > 0) syms.Add(s);
        if (syms.Count == 0) return lens;
        if (syms.Count == 1) { lens[syms[0]] = 1; return lens; }

        var w = new List<long>(); var left = new List<int>(); var right = new List<int>();   // leaf: left=-1, right=symbol
        var pq = new PriorityQueue<int, (long, int)>();
        foreach (int s in syms)
        {
            w.Add(freq[s]); left.Add(-1); right.Add(s);
            pq.Enqueue(w.Count - 1, (freq[s], w.Count - 1));
        }
        while (pq.Count > 1)
        {
            int a = pq.Dequeue(), b = pq.Dequeue();
            w.Add(w[a] + w[b]); left.Add(a); right.Add(b);
            pq.Enqueue(w.Count - 1, (w[^1], w.Count - 1));
        }
        var stack = new Stack<(int node, int depth)>();
        stack.Push((w.Count - 1, 0));
        while (stack.Count > 0)
        {
            var (n, d) = stack.Pop();
            if (left[n] < 0) lens[right[n]] = d;
            else { stack.Push((left[n], d + 1)); stack.Push((right[n], d + 1)); }
        }
        foreach (int s in syms) if (lens[s] > 32) throw new SarcException("E_CAPACITY", "code length > 32 (block too large)");
        return lens;
    }

    private static (ulong code, int len)[] Table(int[] lens)
    {
        var t = new (ulong, int)[lens.Length];
        var items = new List<(int sym, int len)>();
        for (int s = 0; s < lens.Length; s++) if (lens[s] > 0) items.Add((s, lens[s]));
        if (items.Count == 1) { t[items[0].sym] = (0, 1); return t; }
        foreach (var kv in Fmt.CanonCodes(items)) t[kv.Key] = kv.Value;
        return t;
    }

    private static void TreeBytes(int[] lens, List<byte> o)
    {
        var items = new List<(int sym, int len)>();
        for (int s = 0; s < lens.Length; s++) if (lens[s] > 0) items.Add((s, lens[s]));
        items = items.OrderBy(t => t.len).ThenBy(t => t.sym).ToList();
        o.Add((byte)(items.Count >> 8)); o.Add((byte)items.Count);
        foreach (var (s, l) in items) { o.Add((byte)(s >> 8)); o.Add((byte)s); o.Add((byte)l); }
    }

    private sealed class BitWriter
    {
        public readonly List<byte> Buf = new(); private ulong _acc; private int _n;
        public void Put(ulong code, int len)
        {
            _acc = (_acc << len) | code; _n += len;
            while (_n >= 8) { _n -= 8; Buf.Add((byte)(_acc >> _n)); }
            _acc &= (1UL << _n) - 1;
        }
        public byte[] Finish()
        {
            if (_n > 0) { Buf.Add((byte)(_acc << (8 - _n))); _n = 0; _acc = 0; }
            return Buf.ToArray();
        }
    }

    /// <summary>Tokens -> coded block (3 trees, token count, byte count, bitstream). Distance = slot bit_length(dist-1) + (slot-1) extra bits.</summary>
    public static (int[] L, int[] M, int[] D) TokenCodeLengths(List<Tok> toks)
    {
        var lf = new long[257]; var mf = new long[256]; var df = new long[32];
        foreach (var t in toks)
        {
            if (t.Len == 0) lf[t.Lit]++;
            else { lf[256]++; mf[t.Len - 3]++; df[BitLen(t.Dist - 1)]++; }
        }
        return (CodeLengths(lf), CodeLengths(mf), CodeLengths(df));
    }

    public static byte[] EncodeTokens(List<Tok> toks)
    {
        var (L, M, D) = TokenCodeLengths(toks);
        var Lc = Table(L); var Mc = Table(M); var Dc = Table(D);
        var bw = new BitWriter();
        foreach (var t in toks)
        {
            if (t.Len == 0) { bw.Put(Lc[t.Lit].code, Lc[t.Lit].len); continue; }
            bw.Put(Lc[256].code, Lc[256].len);
            bw.Put(Mc[t.Len - 3].code, Mc[t.Len - 3].len);
            int ds = BitLen(t.Dist - 1);
            bw.Put(Dc[ds].code, Dc[ds].len);
            if (ds > 1) bw.Put((ulong)((t.Dist - 1) & ((1 << (ds - 1)) - 1)), ds - 1);
        }
        byte[] body = bw.Finish();
        var o = new List<byte>();
        TreeBytes(L, o); TreeBytes(M, o); TreeBytes(D, o);
        var cnt = new byte[8];
        BinaryPrimitives.WriteUInt32BigEndian(cnt, (uint)toks.Count);
        BinaryPrimitives.WriteUInt32BigEndian(cnt.AsSpan(4), (uint)body.Length);
        o.AddRange(cnt); o.AddRange(body);
        return o.ToArray();
    }
}
