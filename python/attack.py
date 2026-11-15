"""Attack module. (1) GA cryptanalysis of deliberately weak ciphers (XOR short key, Vigenere, substitution) with
n-gram fitness. AES is NOT attacked. (2) Integrity test: flip bytes of an archive and check it is rejected."""
import random, math, os, glob, csv, struct
import numpy as np
from fmt import SarcError

def build_model(train_dir):
    """n-gram model from natural-text/code training files: byte unigram + bigram log-probabilities.
    Returns (uni, bi). The unigram term gives each key byte its own gradient (a pure bigram score is deceptive)."""
    cnt = np.ones((256, 256))
    with open(os.path.join(train_dir, 'labels.csv'), encoding='utf-8') as lf: rows = list(csv.DictReader(lf))
    for row in rows:
        if int(row['type']) in (0, 1):
            with open(os.path.join(train_dir, row['file']), 'rb') as f: a = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
            np.add.at(cnt, (a[:-1], a[1:]), 1)
    bi = np.log(cnt / cnt.sum(axis=1, keepdims=True)); u = cnt.sum(axis=1); uni = np.log(u / u.sum())
    return uni, bi

def score(model, data):
    uni, bi = model
    a = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if len(a) < 2: return -99.0
    return float(uni[a].mean() + bi[a[:-1], a[1:]].mean())

def xor_dec(ct, key): return bytes(c ^ key[i % len(key)] for i, c in enumerate(ct))
def vig_dec(ct, key): return bytes((c - key[i % len(key)]) % 256 for i, c in enumerate(ct))
def subst_dec(ct, perm):
    inv = [0] * 256
    for p, c in enumerate(perm): inv[c] = p
    return bytes(inv[c] for c in ct)

def ga_key(ct, model, dec, klen, pop=40, gens=60, seed=1, sample=2000):
    """GA over key bytes: tournament selection, uniform crossover, point mutation, elitism."""
    r = random.Random(seed); ct = ct[:sample]
    fit = lambda k: score(model, dec(ct, bytes(k)))
    P = [[r.randrange(256) for _ in range(klen)] for _ in range(pop)]; F = [fit(k) for k in P]
    for g in range(gens):
        o = sorted(range(pop), key=lambda i: -F[i]); N = [P[o[0]][:], P[o[1]][:]]; NF = [F[o[0]], F[o[1]]]
        while len(N) < pop:
            a = max(r.sample(range(pop), 3), key=lambda i: F[i]); b = max(r.sample(range(pop), 3), key=lambda i: F[i])
            c = [P[a][i] if r.random() < .5 else P[b][i] for i in range(klen)]
            if r.random() < .5: c[r.randrange(klen)] = r.randrange(256)
            N.append(c); NF.append(fit(c))
        P, F = N, NF
    i = max(range(pop), key=lambda j: F[j]); return bytes(P[i]), F[i]

def crack_repeating_key(ct, model, dec, max_len=8, **kw):
    best = None
    for L in range(1, max_len + 1):
        k, f = ga_key(ct, model, dec, L, **kw)
        if best is None or f > best[1] + 0.02 * (L > 1): best = (k, f, L)       # small penalty for longer keys
    return best

def crack_substitution(ct, model, pop=60, gens=400, seed=3, sample=3000):
    """GA over byte permutations (swap mutation, tournament selection, elitism). Population is seeded from
    frequency-rank matching (cipher symbol rank -> plaintext symbol rank) with random swaps; the GA refines it."""
    r = random.Random(seed); ct = ct[:sample]; uni = model[0]
    fit = lambda p: score(model, subst_dec(ct, p))
    freq = np.bincount(np.frombuffer(ct, dtype=np.uint8), minlength=256)
    cs = sorted(range(256), key=lambda c: (-freq[c], c)); ps = sorted(range(256), key=lambda p: -uni[p])
    base = [0] * 256                      # perm[p] = cipher symbol c  (subst_dec inverts it)
    for c, p in zip(cs, ps): base[p] = c
    present = set(ct)                                # swaps that touch no observed cipher symbol are wasted
    def rswap(p):
        pos = [i for i in range(256) if p[i] in present]
        i = r.choice(pos); j = r.randrange(256); p[i], p[j] = p[j], p[i]
    P = []
    for k in range(pop):
        p = base[:]
        for _ in range(k):
            rswap(p)
        P.append(p)
    F = [fit(p) for p in P]
    for g in range(gens):
        o = sorted(range(pop), key=lambda i: -F[i]); N = [P[o[0]][:], P[o[1]][:]]; NF = [F[o[0]], F[o[1]]]
        while len(N) < pop:
            a_ = max(r.sample(range(pop), 3), key=lambda i: F[i]); c = P[a_][:]
            for _ in range(r.randint(1, 2)):
                rswap(c)
            N.append(c); NF.append(fit(c))
        P, F = N, NF
    i = max(range(pop), key=lambda j: F[j]); return P[i], F[i]

def integrity_attack(archive_bytes, password, open_fn, positions=None, seed=5, n=60):
    """Flip one bit at many positions; every modified archive MUST be rejected. Returns list of (pos, code) accepted/rejected."""
    r = random.Random(seed); L = len(archive_bytes)
    pos = positions or sorted({0, 4, 5, 9, 25, 37, 41, L - 33, L - 32, L - 1} | {r.randrange(L) for _ in range(n)})
    res = []
    for p in pos:
        if not 0 <= p < L: continue
        b = bytearray(archive_bytes); b[p] ^= 0x01
        try: open_fn(bytes(b), password); res.append((p, 'ACCEPTED'))
        except SarcError as e: res.append((p, e.code))
    return res
