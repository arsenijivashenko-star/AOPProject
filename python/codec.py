"""Block codec: parse (greedy|lazy|dp) -> Huffman -> or STORE when no gain (NO_COMPRESSION)."""
import lz, huff
from fmt import decode_block

STORE, GREEDY, LAZY, DP = 0, 1, 2, 3
STRAT_NAMES = {0: 'store', 1: 'greedy', 2: 'lazy', 3: 'dp'}

class Params:
    def __init__(self, window=4096, depth=32, min_match=4, store_threshold=0.90, block_size=16384, strategy=LAZY, lazy_margin=1):
        self.window = window; self.depth = depth; self.min_match = min_match
        self.store_threshold = store_threshold; self.block_size = block_size
        self.strategy = strategy; self.lazy_margin = lazy_margin
    def copy(self, **kw):
        p = Params(**self.__dict__); p.__dict__.update(kw); return p

def encode_block(raw: bytes, p: Params):
    """returns (flag, coded). flag 1 => stored raw (coded == raw)."""
    if len(raw) == 0 or p.strategy == STORE: return 1, raw
    if p.strategy == GREEDY: toks = lz.parse_greedy(raw, p.window, p.depth, p.min_match)
    elif p.strategy == LAZY: toks = lz.parse_lazy(raw, p.window, p.depth, p.min_match, p.lazy_margin)
    else: toks, _ = lz.parse_dp(raw, p.window, p.depth, p.min_match)
    coded = huff.encode_tokens(toks)
    if len(coded) >= len(raw) * p.store_threshold: return 1, raw
    return 0, coded
