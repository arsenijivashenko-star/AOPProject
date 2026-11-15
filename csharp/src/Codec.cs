namespace Sarc;

public static class Strategy
{
    public const int Store = 0, Greedy = 1, Lazy = 2, Dp = 3, Dp2 = 4;
    public static string Name(int s) => s switch { 0 => "store", 1 => "greedy", 2 => "lazy", 3 => "dp", 4 => "dp-iter", _ => "?" };
}

/// <summary>Compression parameters (defaults = best configuration found in the experiments).</summary>
public sealed class CodecParams
{
    public int Window = 4096, Depth = 32, MinMatch = 4, BlockSize = 16384, Strategy = Sarc.Strategy.Lazy, LazyMargin = 1, DpIters = 3;
    public double StoreThreshold = 0.90;
    public CodecParams Clone() => (CodecParams)MemberwiseClone();
}

public static class Codec
{
    /// <summary>Returns (flag, coded). flag 1 = stored raw (NO_COMPRESSION, coded == raw).</summary>
    public static (byte flag, byte[] coded) EncodeBlock(byte[] raw, CodecParams p)
    {
        if (raw.Length == 0 || p.Strategy == Strategy.Store) return (1, raw);
        List<Tok> toks = p.Strategy switch
        {
            Strategy.Greedy => Lz.ParseGreedy(raw, p.Window, p.Depth, p.MinMatch),
            Strategy.Lazy => Lz.ParseLazy(raw, p.Window, p.Depth, p.MinMatch, p.LazyMargin),
            Strategy.Dp2 => ParseDpIter(raw, p.Window, p.Depth, p.MinMatch, p.DpIters).toks,
            _ => Lz.ParseDp(raw, p.Window, p.Depth, p.MinMatch).toks,
        };
        byte[] coded = Huffman.EncodeTokens(toks);
        if (coded.Length >= raw.Length * p.StoreThreshold) return (1, raw);
        return (0, coded);
    }

    /// <summary>Per-symbol costs (bits) from the Huffman lengths of a previous parse. A symbol that was absent gets max length + 2.</summary>
    private static int[] CostArr(int[] lens, int n)
    {
        int worst = 1;
        foreach (int x in lens) if (x > worst) worst = x;
        worst += 2;
        var r = new int[n];
        for (int s = 0; s < n; s++) r[s] = s < lens.Length && lens[s] > 0 ? lens[s] : worst;
        return r;
    }

    /// <summary>Iterative two-pass parse (expert level). Pass 0 = DP with the FIXED table from team_config.json; every next pass
    /// uses the Huffman lengths of the previous parse as FIXED costs. The real coded size is measured after each pass and the best
    /// parse is kept, so the result is never worse than pass 0. hist[i] = coded bytes after pass i.</summary>
    public static (List<Tok> toks, List<int> hist) ParseDpIter(byte[] raw, int window, int depth, int minMatch, int iters)
    {
        var cand = Lz.CandidatesAll(raw, window, depth, minMatch);
        List<Tok> best = Lz.ParseDp(raw, window, depth, minMatch, cand).toks;
        int bestSize = Huffman.EncodeTokens(best).Length;
        var hist = new List<int> { bestSize };
        List<Tok> cur = best;
        for (int k = 0; k < iters; k++)
        {
            var (L, M, D) = Huffman.TokenCodeLengths(cur);
            int[] lit = CostArr(L, 257), len = CostArr(M, 256), dist = CostArr(D, 32);
            cur = Lz.ParseDpCosts(raw, lit[..256], len, dist, lit[256], window, depth, minMatch, cand).toks;
            int size = Huffman.EncodeTokens(cur).Length;
            hist.Add(size);
            if (size < bestSize) { best = cur; bestSize = size; }
        }
        return (best, hist);
    }
}
