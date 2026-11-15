namespace Sarc;

public static class Strategy
{
    public const int Store = 0, Greedy = 1, Lazy = 2, Dp = 3;
    public static string Name(int s) => s switch { 0 => "store", 1 => "greedy", 2 => "lazy", 3 => "dp", _ => "?" };
}

/// <summary>Compression parameters (defaults = best configuration found in the experiments).</summary>
public sealed class CodecParams
{
    public int Window = 4096, Depth = 32, MinMatch = 4, BlockSize = 16384, Strategy = Sarc.Strategy.Lazy, LazyMargin = 1;
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
            _ => Lz.ParseDp(raw, p.Window, p.Depth, p.MinMatch).toks,
        };
        byte[] coded = Huffman.EncodeTokens(toks);
        if (coded.Length >= raw.Length * p.StoreThreshold) return (1, raw);
        return (0, coded);
    }
}
