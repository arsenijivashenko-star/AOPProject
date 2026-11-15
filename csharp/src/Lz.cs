namespace Sarc;

/// <summary>LZ77 parsers written from scratch: greedy, lazy, optimal (dynamic programming with fixed costs).
/// Matches never leave the block (the history window is reset per block).</summary>
public static class Lz
{
    public const int MinMatch = 3, MaxMatch = 258, MaxWindow = 32768;
    public const int LitCost = 9;                                   // team_config: fixed DP costs
    public static int RefCost(int dist) => 17 + Math.Max(0, Huffman.BitLen(dist - 1) - 1);

    /// <summary>Hash chains over 3-byte prefixes. Candidates are returned nearest-first with strictly growing length.</summary>
    public sealed class MatchFinder
    {
        private readonly byte[] _d; private readonly int _n, _window, _depth, _minm;
        private readonly int[] _head = new int[1 << 16]; private readonly int[] _prev; private int _inserted;

        public MatchFinder(byte[] data, int window, int depth, int minMatch)
        {
            _d = data; _n = data.Length; _window = window; _depth = depth; _minm = Math.Max(MinMatch, minMatch);
            _prev = new int[Math.Max(1, _n)];
            Array.Fill(_head, -1);
        }

        private static int Hash(byte[] d, int p) => ((d[p] << 8) ^ (d[p + 1] << 4) ^ d[p + 2] ^ (d[p + 2] << 11)) & 0xFFFF;

        private void InsertUpTo(int i)
        {
            while (_inserted < i && _inserted + 3 <= _n)
            {
                int p = _inserted, h = Hash(_d, p);
                _prev[p] = _head[h]; _head[h] = p;
                _inserted++;
            }
            if (_inserted < i) _inserted = i;
        }

        public List<(int len, int dist)> Candidates(int i)
        {
            var res = new List<(int len, int dist)>();
            if (i + _minm > _n) return res;
            InsertUpTo(i);
            int maxl = Math.Min(MaxMatch, _n - i);
            int j = _head[Hash(_d, i)], best = _minm - 1, steps = 0, lo = i - _window;
            while (j >= 0 && j >= lo && steps < _depth)
            {
                if (best < maxl && _d[j + best] == _d[i + best])
                {
                    int l = new ReadOnlySpan<byte>(_d, j, maxl).CommonPrefixLength(new ReadOnlySpan<byte>(_d, i, maxl));
                    if (l > best)
                    {
                        best = l; res.Add((l, i - j));
                        if (l >= maxl) break;
                    }
                }
                j = _prev[j]; steps++;
            }
            return res;
        }
    }

    public static List<Tok> ParseGreedy(byte[] data, int window, int depth, int minMatch)
    {
        var mf = new MatchFinder(data, window, depth, minMatch); var toks = new List<Tok>(); int i = 0, n = data.Length;
        while (i < n)
        {
            var c = mf.Candidates(i);
            if (c.Count > 0) { var (l, d) = c[^1]; toks.Add(new Tok(l, d, 0)); i += l; }
            else { toks.Add(new Tok(0, 0, data[i])); i++; }
        }
        return toks;
    }

    /// <summary>Lazy matching: if the match at i+1 is longer by at least <paramref name="margin"/>, emit a literal at i.</summary>
    public static List<Tok> ParseLazy(byte[] data, int window, int depth, int minMatch, int margin)
    {
        var mf = new MatchFinder(data, window, depth, minMatch); var toks = new List<Tok>(); int i = 0, n = data.Length;
        var cur = n > 0 ? mf.Candidates(0) : new List<(int len, int dist)>();
        while (i < n)
        {
            if (cur.Count > 0)
            {
                var (l, d) = cur[^1];
                var nxt = i + 1 < n ? mf.Candidates(i + 1) : new List<(int len, int dist)>();
                if (nxt.Count > 0 && nxt[^1].len >= l + margin)
                {
                    toks.Add(new Tok(0, 0, data[i])); i++; cur = nxt; continue;
                }
                toks.Add(new Tok(l, d, 0)); i += l;
                cur = i < n ? mf.Candidates(i) : new List<(int len, int dist)>();
            }
            else
            {
                toks.Add(new Tok(0, 0, data[i])); i++;
                cur = i < n ? mf.Candidates(i) : new List<(int len, int dist)>();
            }
        }
        return toks;
    }

    /// <summary>Candidate lists of every position (forward pass: the finder inserts positions incrementally).</summary>
    public static List<(int len, int dist)>[] CandidatesAll(byte[] data, int window, int depth, int minMatch)
    {
        var mf = new MatchFinder(data, window, depth, minMatch);
        var cand = new List<(int len, int dist)>[data.Length];
        for (int i = 0; i < data.Length; i++) cand[i] = mf.Candidates(i);
        return cand;
    }

    private static List<Tok> Backtrack(byte[] data, int[] chLen, int[] chDist)
    {
        var toks = new List<Tok>();
        for (int i = 0; i < data.Length;)
        {
            if (chLen[i] == 0) { toks.Add(new Tok(0, 0, data[i])); i++; }
            else { toks.Add(new Tok(chLen[i], chDist[i], 0)); i += chLen[i]; }
        }
        return toks;
    }

    /// <summary>Optimal parse. C[i] = min( lit + C[i+1], min over (d,l) in M_i of cost(d,l) + C[i+l] ), computed from the end of the
    /// block backwards, with FIXED costs (9 bits literal, 17 + max(0,bitlen(d-1)-1) bits reference). O(sum |M_i|).</summary>
    public static (List<Tok> toks, long cost) ParseDp(byte[] data, int window, int depth, int minMatch,
        List<(int len, int dist)>[]? cand = null)
    {
        int n = data.Length;
        if (n == 0) return (new List<Tok>(), 0);
        cand ??= CandidatesAll(data, window, depth, minMatch);
        var C = new long[n + 1]; var chLen = new int[n]; var chDist = new int[n];
        int minm = Math.Max(MinMatch, minMatch);
        for (int i = n - 1; i >= 0; i--)
        {
            long best = LitCost + C[i + 1]; int bl = 0, bd = 0;
            int prevl = minm - 1;
            foreach (var (l, dist) in cand[i])
            {
                long rc = RefCost(dist);                                  // lengths prevl+1..l are all reachable at this distance
                for (int len = prevl + 1; len <= l; len++)
                {
                    long v = rc + C[i + len];
                    if (v < best) { best = v; bl = len; bd = dist; }
                }
                prevl = l;
            }
            C[i] = best; chLen[i] = bl; chDist[i] = bd;
        }
        return (Backtrack(data, chLen, chDist), C[0]);
    }

    /// <summary>Optimal parse with per-symbol costs taken from Huffman code lengths of a PREVIOUS pass (constant during this pass, so
    /// the recursion is well defined). litBits[b]: literal b; lenBits[l-3]: length symbol; distBits[slot]: distance slot (the slot-1
    /// extra bits are added here); escBits: marker 256 in the literal tree. The reference cost depends on the length too.</summary>
    public static (List<Tok> toks, long cost) ParseDpCosts(byte[] data, int[] litBits, int[] lenBits, int[] distBits, int escBits,
        int window, int depth, int minMatch, List<(int len, int dist)>[]? cand = null)
    {
        int n = data.Length;
        if (n == 0) return (new List<Tok>(), 0);
        cand ??= CandidatesAll(data, window, depth, minMatch);
        var C = new long[n + 1]; var chLen = new int[n]; var chDist = new int[n];
        int minm = Math.Max(MinMatch, minMatch);
        for (int i = n - 1; i >= 0; i--)
        {
            long best = litBits[data[i]] + C[i + 1]; int bl = 0, bd = 0;
            int prevl = minm - 1;
            foreach (var (l, dist) in cand[i])
            {
                int ds = Huffman.BitLen(dist - 1);
                long baseCost = escBits + distBits[ds] + Math.Max(0, ds - 1);
                for (int len = prevl + 1; len <= l; len++)
                {
                    long v = baseCost + lenBits[len - 3] + C[i + len];
                    if (v < best) { best = v; bl = len; bd = dist; }
                }
                prevl = l;
            }
            C[i] = best; chLen[i] = bl; chDist[i] = bd;
        }
        return (Backtrack(data, chLen, chDist), C[0]);
    }

    public static long TokensCost(List<Tok> toks)
    {
        long s = 0;
        foreach (var t in toks) s += t.Len == 0 ? LitCost : RefCost(t.Dist);
        return s;
    }

    public static byte[] Expand(List<Tok> toks)
    {
        var o = new ByteBuf();
        foreach (var t in toks)
        {
            if (t.Len == 0) o.Add(t.Lit);
            else for (int k = 0; k < t.Len; k++) o.Add(o.A[o.N - t.Dist]);
        }
        return o.ToArray();
    }
}
