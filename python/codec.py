"""Block codec: parse (greedy|lazy|dp) -> Huffman -> or STORE when no gain (NO_COMPRESSION)."""
import lz, huff
from fmt import decode_block

STORE, GREEDY, LAZY, DP, DP2 = 0, 1, 2, 3, 4
STRAT_NAMES = {0: 'store', 1: 'greedy', 2: 'lazy', 3: 'dp', 4: 'dp-iter'}

class Params:
    def __init__(self, window=4096, depth=32, min_match=4, store_threshold=0.90, block_size=16384, strategy=LAZY, lazy_margin=1, dp_iters=3):
        self.window = window; self.depth = depth; self.min_match = min_match
        self.store_threshold = store_threshold; self.block_size = block_size
        self.strategy = strategy; self.lazy_margin = lazy_margin; self.dp_iters = dp_iters
    def copy(self, **kw):
        p = Params(**self.__dict__); p.__dict__.update(kw); return p

def encode_block(raw: bytes, p: Params):
    """returns (flag, coded). flag 1 => stored raw (coded == raw)."""
    if len(raw) == 0 or p.strategy == STORE: return 1, raw
    if p.strategy == GREEDY: toks = lz.parse_greedy(raw, p.window, p.depth, p.min_match)
    elif p.strategy == LAZY: toks = lz.parse_lazy(raw, p.window, p.depth, p.min_match, p.lazy_margin)
    elif p.strategy == DP2: toks = parse_dp_iter(raw, p.window, p.depth, p.min_match, p.dp_iters)[0]
    else: toks, _ = lz.parse_dp(raw, p.window, p.depth, p.min_match)
    coded = huff.encode_tokens(toks)
    if len(coded) >= len(raw) * p.store_threshold: return 1, raw
    return 0, coded

def _costs_from_tokens(toks):
    """Per-symbol costs (bits) from the Huffman lengths of a previous parse. Unseen symbols get max_len + 2 as an estimate
    (they would be added to the rebuilt tree with a long code)."""
    L, M, D = huff.token_code_lengths(toks)
    def arr(lens, n):
        worst = (max(lens.values()) if lens else 1) + 2
        return [lens.get(s, worst) for s in range(n)]
    lit = arr(L, 257); return lit[:256], arr(M, 256), arr(D, 17), lit[256]

def parse_dp_iter(raw, window, depth, min_match, iters=3):
    """Iterative two-pass scheme (expert level): pass 0 = DP with the FIXED table from team_config.json, then repeat
    {build canonical Huffman lengths from the current parse -> use them as FIXED costs -> DP again}.
    Costs are constant inside every pass, so the recursion stays well defined. After every pass the REAL coded size is
    measured; the best parse seen so far is kept, so the result is never worse than pass 0 (monotone by construction).
    Returns (tokens, history) where history[i] = coded bytes after pass i."""
    cand = lz._candidates_all(raw, window, depth, min_match)
    toks, _ = lz.parse_dp(raw, window, depth, min_match, cand=cand)
    best = toks; best_sz = len(huff.encode_tokens(toks)); hist = [best_sz]
    cur = toks
    for _ in range(iters):
        lit, ln, ds, esc = _costs_from_tokens(cur)
        cur, _ = lz.parse_dp_costs(raw, lit, ln, ds, esc, window, depth, min_match, cand=cand)
        sz = len(huff.encode_tokens(cur)); hist.append(sz)
        if sz < best_sz: best, best_sz = cur, sz
    return best, hist
