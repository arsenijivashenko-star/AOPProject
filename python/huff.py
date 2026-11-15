"""Huffman coding from scratch: heap-based tree -> code LENGTHS -> canonical codes (spec: only lengths are stored)."""
import heapq, struct
from fmt import canon_codes

def code_lengths(freq):
    """freq: {sym: count>0}. Returns {sym: length}. Single symbol -> length 1 (degenerate alphabet)."""
    syms = [s for s, c in freq.items() if c > 0]
    if not syms: return {}
    if len(syms) == 1: return {syms[0]: 1}
    # heap items: (weight, tiebreak, [symbols]); merging increments depth of every contained symbol
    heap = [(freq[s], s, [s]) for s in syms]; heapq.heapify(heap)
    depth = {s: 0 for s in syms}; uid = 1 << 20
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap); w2, _, b = heapq.heappop(heap)
        for s in a: depth[s] += 1
        for s in b: depth[s] += 1
        heapq.heappush(heap, (w1 + w2, uid, a + b)); uid += 1
    return depth

def tree_bytes(lens):
    items = sorted(lens.items(), key=lambda t: (t[1], t[0]))
    return struct.pack('>H', len(items)) + b''.join(struct.pack('>HB', s, l) for s, l in items)

class BitWriter:
    def __init__(self): self.buf = bytearray(); self.acc = 0; self.n = 0
    def put(self, code, length):
        self.acc = (self.acc << length) | code; self.n += length
        while self.n >= 8:
            self.n -= 8; self.buf.append((self.acc >> self.n) & 0xff)
        self.acc &= (1 << self.n) - 1
    def finish(self):
        if self.n: self.buf.append((self.acc << (8 - self.n)) & 0xff); self.n = 0
        return bytes(self.buf)

def token_code_lengths(toks):
    """Huffman code lengths of the three trees (literal+marker 256 / length-3 / distance slot) for a token list."""
    lf = {}; mf = {}; df = {}
    for t in toks:
        if t[0] == 'L': lf[t[1]] = lf.get(t[1], 0) + 1
        else:
            lf[256] = lf.get(256, 0) + 1
            mf[t[1] - 3] = mf.get(t[1] - 3, 0) + 1
            ds = (t[2] - 1).bit_length(); df[ds] = df.get(ds, 0) + 1
    return code_lengths(lf), code_lengths(mf), code_lengths(df)

def encode_tokens(toks):
    """tokens -> coded block body (3 trees + counts + bitstream). Returns bytes."""
    L, M, D = token_code_lengths(toks)
    def table(lens):
        if len(lens) == 1: return {s: (0, 1) for s in lens}
        return canon_codes(list(lens.items()))
    Lc, Mc, Dc = table(L), table(M), table(D)
    bw = BitWriter()
    for t in toks:
        if t[0] == 'L': bw.put(*Lc[t[1]])
        else:
            bw.put(*Lc[256]); bw.put(*Mc[t[1] - 3])
            dist = t[2]; ds = (dist - 1).bit_length(); bw.put(*Dc[ds])
            if ds > 1: bw.put((dist - 1) & ((1 << (ds - 1)) - 1), ds - 1)
    body = bw.finish()
    return tree_bytes(L) + tree_bytes(M) + tree_bytes(D) + struct.pack('>II', len(toks), len(body)) + body

def entropy_bits(freq):
    import math
    n = sum(freq.values()); return -sum(c * math.log2(c / n) for c in freq.values() if c)
