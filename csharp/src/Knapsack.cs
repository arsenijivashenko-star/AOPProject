namespace Sarc;

/// <summary>0/1 knapsack: exact DP O(n*W) and the greedy value/weight heuristic (for volume packing).</summary>
public static class Knapsack
{
    public static List<int> Dp(IReadOnlyList<long> values, IReadOnlyList<int> weights, int W)
    {
        int n = values.Count;
        var chosen = new List<int>();
        if (W <= 0 || n == 0) return chosen;
        var dp = new long[W + 1];
        int words = (W + 64) / 64;
        var keep = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            keep[i] = new ulong[words];
            int w = weights[i];
            if (w > W) continue;
            long v = values[i];
            for (int c = W; c >= w; c--)                              // descending c => each item used at most once
            {
                long cand = dp[c - w] + v;
                if (cand > dp[c]) { dp[c] = cand; keep[i][c >> 6] |= 1UL << (c & 63); }
            }
        }
        int cap = 0;                                                  // smallest capacity reaching the maximum
        for (int c = 1; c <= W; c++) if (dp[c] > dp[cap]) cap = c;
        for (int i = n - 1; i >= 0; i--)
            if ((keep[i][cap >> 6] >> (cap & 63) & 1UL) != 0) { chosen.Add(i); cap -= weights[i]; }
        chosen.Sort();
        return chosen;
    }

    public static List<int> Greedy(IReadOnlyList<long> values, IReadOnlyList<int> weights, int W)
    {
        var order = Enumerable.Range(0, values.Count)
            .OrderByDescending(i => (double)values[i] / Math.Max(1, weights[i])).ToList();
        var chosen = new List<int>(); long used = 0;
        foreach (int i in order) if (used + weights[i] <= W) { chosen.Add(i); used += weights[i]; }
        chosen.Sort();
        return chosen;
    }

    public static long Total(IReadOnlyList<long> values, IEnumerable<int> idx) => idx.Sum(i => values[i]);
}
