"""LZ77 parsers written from scratch: greedy, lazy, optimal (DP, fixed costs).
Tokens: ('L', byte) or ('M', length, dist).  Matches never leave the block (window reset per block)."""
import numpy as np

MIN_MATCH = 3
MAX_MATCH = 258          # length symbol = len-3 in 0..255
MAX_WINDOW = 32768

def match_len(d, j, i, maxl):
    """length of common prefix of d[j:] and d[i:] bounded by maxl (j<i)."""
    n = 0
    while n + 16 <= maxl and d[j + n:j + n + 16] == d[i + n:i + n + 16]:
        n += 16
    while n < maxl and d[j + n] == d[i + n]:
        n += 1
    return n

class MatchFinder:
    """Hash chains over 3-byte prefixes. Candidates are returned nearest-first."""
    def __init__(self, data, window=MAX_WINDOW, depth=32, min_match=MIN_MATCH):
        self.d = data; self.n = len(data); self.window = window
        self.depth = depth; self.minm = max(MIN_MATCH, min_match)
        self.head = {}; self.prev = [-1] * self.n; self.inserted = 0

    def _insert_upto(self, i):
        d = self.d; head = self.head; prev = self.prev
        while self.inserted < i and self.inserted + 3 <= self.n:
            p = self.inserted
            k = d[p:p + 3]
            prev[p] = head.get(k, -1); head[k] = p
            self.inserted += 1
        self.inserted = max(self.inserted, i)

    def candidates(self, i):
        """list of (length, dist) with strictly increasing length as dist grows (Pareto: nearest for each length)."""
        if i + self.minm > self.n: return []
        self._insert_upto(i)
        d = self.d; maxl = min(MAX_MATCH, self.n - i)
        j = self.head.get(d[i:i + 3], -1); out = []; best = self.minm - 1; steps = 0
        lo = i - self.window
        while j >= 0 and j >= lo and steps < self.depth:
            if d[j + best] == d[i + best] if best < maxl else False:
                l = match_len(d, j, i, maxl)
                if l > best:
                    best = l; out.append((l, i - j))
                    if l >= maxl: break
            j = self.prev[j]; steps += 1
        return out

def parse_greedy(data, window=MAX_WINDOW, depth=32, min_match=MIN_MATCH):
    mf = MatchFinder(data, window, depth, min_match); toks = []; i = 0; n = len(data)
    while i < n:
        c = mf.candidates(i)
        if c:
            l, dist = c[-1]; toks.append(('M', l, dist)); i += l
        else:
            toks.append(('L', data[i])); i += 1
    return toks

def parse_lazy(data, window=MAX_WINDOW, depth=32, min_match=MIN_MATCH, margin=1):
    """Lazy matching: if the match at i+1 is longer by >= margin, emit literal at i."""
    mf = MatchFinder(data, window, depth, min_match); toks = []; i = 0; n = len(data)
    cur = mf.candidates(0) if n else []
    while i < n:
        if cur:
            l, dist = cur[-1]
            nxt = mf.candidates(i + 1) if i + 1 < n else []
            if nxt and nxt[-1][0] >= l + margin:
                toks.append(('L', data[i])); i += 1; cur = nxt; continue
            toks.append(('M', l, dist)); i += l
            cur = mf.candidates(i) if i < n else []
        else:
            toks.append(('L', data[i])); i += 1
            cur = mf.candidates(i) if i < n else []
    return toks

# ---- fixed DP costs from team_config.json ----
LIT_COST = 9
def ref_cost(dist):
    return 17 + max(0, (dist - 1).bit_length() - 1)

def parse_dp(data, window=MAX_WINDOW, depth=32, min_match=MIN_MATCH, costs=None):
    """Optimal parsing. C[i] = min(lit + C[i+1], min_{(d,l) in M_i} cost(d,l) + C[i+l]); computed from the end.
    Candidate set per position: for every length l the NEAREST distance (cheapest extra bits)."""
    n = len(data)
    if n == 0: return []
    lit, refc = costs if costs else (LIT_COST, ref_cost)
    mf = MatchFinder(data, window, depth, min_match)
    cand = [None] * n
    for i in range(n): cand[i] = mf.candidates(i)      # forward pass: finder needs ascending insertion
    C = np.zeros(n + 1, dtype=np.int64); choice = [None] * n
    for i in range(n - 1, -1, -1):
        best = lit + C[i + 1]; ch = None
        c = cand[i]
        if c:
            prevl = max(MIN_MATCH, min_match) - 1
            for (l, dist) in c:                       # segment of lengths prevl+1..l all at distance dist
                lo = prevl + 1
                seg = C[i + lo:i + l + 1]             # C[i+len] for len in lo..l
                k = int(np.argmin(seg)); v = refc(dist) + seg[k]
                if v < best: best = v; ch = (lo + k, dist)
                prevl = l
        C[i] = best; choice[i] = ch
    toks = []; i = 0
    while i < n:
        ch = choice[i]
        if ch is None: toks.append(('L', data[i])); i += 1
        else: toks.append(('M', ch[0], ch[1])); i += ch[0]
    return toks, int(C[0])

def tokens_cost(toks):
    return sum(LIT_COST if t[0] == 'L' else ref_cost(t[2]) for t in toks)

def expand(toks):
    out = bytearray()
    for t in toks:
        if t[0] == 'L': out.append(t[1])
        else:
            for _ in range(t[1]): out.append(out[-t[2]])
    return bytes(out)
