"""Genetic algorithm for parameter tuning (fitness = Q), plus random-search and local-search baselines with the SAME
number of fitness evaluations. Full chromosome: strategy, window, depth, min_match, block_size, store_threshold, lazy_margin."""
import random, json, time, copy
from codec import Params, GREEDY, LAZY, DP
import metric
from archive import Config

GENES = {
    'strategy':        [GREEDY, LAZY, DP],
    'window':          [1024, 2048, 4096, 8192, 16384, 32768],
    'depth':           [1, 2, 4, 8, 16, 32, 64],
    'min_match':       [3, 4, 5, 6],
    'block_size':      [4096, 8192, 12288, 16384],
    'store_threshold': [0.90, 0.95, 0.98, 1.00],
    'lazy_margin':     [1, 2, 3],
}
KEYS = list(GENES)

def rand_ind(r): return {k: r.choice(v) for k, v in GENES.items()}
def to_cfg(ind, classifier=None, packer='dp'):
    return Config(params=Params(**ind), classifier=classifier, packer=packer)

class Evaluator:
    def __init__(self, src_dir, classifier=None):
        self.src = src_dir; self.clf = classifier; self.n = 0; self.cache = {}; self.points = []
    def __call__(self, ind):
        key = tuple(ind[k] for k in KEYS)
        if key in self.cache: return self.cache[key]       # deterministic caching keeps elitism exact
        m = metric.measure(self.src, to_cfg(ind, self.clf)); self.n += 1
        self.cache[key] = m['Q']; self.points.append((m['S'], m['T'], dict(ind)))
        return m['Q']

def tournament(pop, fit, r, k=3):
    return max(r.sample(range(len(pop)), k), key=lambda i: fit[i])

def cross_uniform(a, b, r): return {k: (a[k] if r.random() < .5 else b[k]) for k in KEYS}
def cross_onepoint(a, b, r):
    c = r.randint(1, len(KEYS) - 1); return {k: (a[k] if i < c else b[k]) for i, k in enumerate(KEYS)}
def mutate(ind, pm, r):
    o = dict(ind)
    for k in KEYS:
        if r.random() < pm: o[k] = r.choice(GENES[k])
    return o

def run_ga(ev, budget, pop_size=10, seed=2001, elite=2, log=None):
    r = random.Random(seed); pop = [rand_ind(r) for _ in range(pop_size)]
    fit = [ev(i) for i in pop]; hist = []; gen = 0; pm = 0.25; best_prev = max(fit)
    def rec(): hist.append(dict(gen=gen, best=max(fit), mean=sum(fit) / len(fit), evals=ev.n, pm=pm))
    rec(); stall = 0
    while ev.n < budget:
        gen += 1
        order = sorted(range(len(pop)), key=lambda i: -fit[i])
        new = [copy.deepcopy(pop[i]) for i in order[:elite]]; nf = [fit[i] for i in order[:elite]]   # elitism: fitness carried unchanged
        while len(new) < pop_size and ev.n < budget + pop_size:
            a = pop[tournament(pop, fit, r)]; b = pop[tournament(pop, fit, r)]
            c = (cross_uniform if r.random() < .5 else cross_onepoint)(a, b, r)        # two crossover types
            c = mutate(c, pm, r); new.append(c); nf.append(ev(c))
            if ev.n >= budget: break
        pop = new; fit = nf
        pop = pop[:len(fit)]
        if max(fit) <= best_prev + 1e-12: stall += 1
        else: stall = 0
        best_prev = max(best_prev, max(fit))
        pm = min(0.6, 0.25 + 0.1 * stall) if stall else 0.25                 # adaptive mutation: grows while stagnating
        rec()
        if len(pop) < 2: break
    i = max(range(len(pop)), key=lambda j: fit[j])
    return pop[i], fit[i], hist

def run_random(ev, budget, seed=7):
    r = random.Random(seed); best = (None, -1); hist = []
    while ev.n < budget:
        i = rand_ind(r); q = ev(i)
        if q > best[1]: best = (i, q)
        hist.append(dict(evals=ev.n, best=best[1]))
    return best[0], best[1], hist

def run_local(ev, budget, seed=11):
    """Hill climbing with random restarts: change one gene at a time, accept improvements."""
    r = random.Random(seed); cur = rand_ind(r); cq = ev(cur); best = (cur, cq); hist = []
    while ev.n < budget:
        k = r.choice(KEYS); cand = dict(cur); cand[k] = r.choice(GENES[k])
        q = ev(cand)
        if q > cq: cur, cq = cand, q
        if cq > best[1]: best = (cur, cq)
        hist.append(dict(evals=ev.n, best=best[1]))
    return best[0], best[1], hist

def pareto(points):
    """points: (S,T,ind). Non-dominated set maximising both S and T."""
    front = []
    for s, t, ind in points:
        if not any((s2 >= s and t2 >= t) and (s2 > s or t2 > t) for s2, t2, _ in points): front.append((s, t, ind))
    seen = set(); out = []
    for s, t, ind in sorted(front, key=lambda x: -x[0]):
        if (round(s, 9), round(t, 9)) not in seen: seen.add((round(s, 9), round(t, 9))); out.append((s, t, ind))
    return out
