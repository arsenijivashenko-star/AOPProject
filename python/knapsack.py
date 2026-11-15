"""0/1 knapsack: exact DP O(n*W) and greedy by value/weight, for volume packing."""
import numpy as np

def knapsack_dp(values, weights, W):
    n = len(values)
    if W <= 0 or n == 0: return []
    dp = np.zeros(W + 1, dtype=np.int64); keep = np.zeros((n, W + 1), dtype=bool)
    for i in range(n):
        w = weights[i]
        if w > W: continue
        cand = dp[:W + 1 - w] + values[i]          # value if item i taken, capacity c -> c+w
        better = cand > dp[w:]
        keep[i, w:] = better
        dp[w:] = np.where(better, cand, dp[w:])
    chosen = []; c = int(np.argmax(dp))               # dp is non-decreasing in c; argmax -> smallest c with max value
    for i in range(n - 1, -1, -1):
        if keep[i, c]: chosen.append(i); c -= weights[i]
    return sorted(chosen)

def knapsack_greedy(values, weights, W):
    order = sorted(range(len(values)), key=lambda i: values[i] / max(1, weights[i]), reverse=True)
    chosen = []; used = 0
    for i in order:
        if used + weights[i] <= W: chosen.append(i); used += weights[i]
    return sorted(chosen)

def total(values, idx): return sum(values[i] for i in idx)
