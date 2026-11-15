"""File-type classifier: multilayer perceptron written from scratch (numpy matrix maths only, no ML library).
Features: Shannon entropy, 32-bin folded byte histogram, printable ASCII share, mean distance between '\\n',
autocorrelation lags 1..16, bigram entropy, plus a few cheap structure features."""
import math, json, os, csv
import numpy as np

CLASSES = ['природний текст', 'вихідний код', 'структуровані дані', 'растр без стиснення', 'виконуваний файл', 'вже стиснуте', 'штучні дані']
SAMPLE = 16384

def features(data: bytes):
    d = data[:SAMPLE]; n = len(d)
    if n == 0: return np.zeros(NFEAT)
    a = np.frombuffer(d, dtype=np.uint8)
    cnt = np.bincount(a, minlength=256).astype(np.float64); p = cnt / n
    nz = p[p > 0]; H = float(-(nz * np.log2(nz)).sum()) / 8.0
    hist32 = p.reshape(32, 8).sum(axis=1)
    printable = float(((a >= 32) & (a < 127)).sum() + (a == 10).sum() + (a == 9).sum()) / n
    nl = np.flatnonzero(a == 10)
    nlgap = float(np.diff(nl).mean()) if len(nl) > 1 else 0.0
    nlgap = min(nlgap, 256.0) / 256.0
    nlfrac = len(nl) / n
    ac = []
    x = a.astype(np.float64) - a.mean(); var = float((x * x).sum())
    for lag in range(1, 17):
        ac.append(float((x[:-lag] * x[lag:]).sum()) / var if var > 0 and n > lag else 0.0)
    if n > 1:
        bg = a[:-1].astype(np.int64) * 256 + a[1:]
        bc = np.bincount(bg, minlength=65536).astype(np.float64); bp = bc[bc > 0] / (n - 1)
        Hb = float(-(bp * np.log2(bp)).sum()) / 16.0
    else: Hb = 0.0
    ctrl = float((a < 9).sum()) / n; high = float((a >= 128).sum()) / n; zero = float((a == 0).sum()) / n
    uniq = len(nz) / 256.0
    alpha = float(((a >= 65) & (a <= 90)).sum() + ((a >= 97) & (a <= 122)).sum()) / n
    punct = float(sum(((a == c).sum() for c in b'(){}[]=:;,"_#<>'))) / n
    comma = float((a == 44).sum()) / n
    sp = float((a == 32).sum()) / n
    zero_run = 0.0
    return np.concatenate([[H, Hb, printable, nlgap, nlfrac, ctrl, high, zero, uniq, alpha, punct, comma, sp], ac, hist32])
NFEAT = 13 + 16 + 32

class MLP:
    def __init__(self, sizes, seed=0):
        r = np.random.RandomState(seed); self.W = []; self.b = []
        for i in range(len(sizes) - 1):
            self.W.append(r.randn(sizes[i], sizes[i + 1]) * math.sqrt(2.0 / sizes[i])); self.b.append(np.zeros(sizes[i + 1]))
    def forward(self, X):
        acts = [X]
        for i in range(len(self.W)):
            z = acts[-1] @ self.W[i] + self.b[i]
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        z = acts[-1] - acts[-1].max(axis=1, keepdims=True); e = np.exp(z)
        return acts, e / e.sum(axis=1, keepdims=True)
    def predict(self, X): return self.forward(X)[1].argmax(axis=1)
    def fit(self, X, y, epochs=400, lr=0.01, l2=1e-4, batch=32, seed=0):
        """Mini-batch Adam, softmax cross-entropy, back-propagation written by hand."""
        r = np.random.RandomState(seed); n = len(X); K = self.W[-1].shape[1]
        Y = np.eye(K)[y]
        m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]; v = [np.zeros_like(q) for q in m]; t = 0
        for ep in range(epochs):
            perm = r.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]; acts, P = self.forward(X[idx])
                g = (P - Y[idx]) / len(idx); gw = []; gb = []
                for i in range(len(self.W) - 1, -1, -1):
                    gw.append(acts[i].T @ g + l2 * self.W[i]); gb.append(g.sum(axis=0))
                    if i: g = (g @ self.W[i].T) * (acts[i] > 0)
                grads = gw[::-1] + gb[::-1]; params = self.W + self.b; t += 1
                for k, (pk, gk) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk
                    pk -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    def to_json(self): return dict(W=[w.tolist() for w in self.W], b=[b.tolist() for b in self.b])
    @staticmethod
    def from_json(j):
        m = MLP([1, 1]); m.W = [np.array(w) for w in j['W']]; m.b = [np.array(b) for b in j['b']]; return m

def macro_f1(y, p, K=7):
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(y, p): cm[a, b] += 1
    f = []
    for k in range(K):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), cm

def load_split(d):
    X = []; y = []
    for row in csv.DictReader(open(os.path.join(d, 'labels.csv'), encoding='utf-8')):
        X.append(features(open(os.path.join(d, row['file']), 'rb').read())); y.append(int(row['type']))
    return np.array(X), np.array(y)

class Classifier:
    def __init__(self, path):
        j = json.load(open(path)); self.mu = np.array(j['mu']); self.sd = np.array(j['sd']); self.net = MLP.from_json(j['net'])
    def __call__(self, data: bytes) -> int:
        x = (features(data) - self.mu) / self.sd
        return int(self.net.predict(x[None, :])[0])

def train(root, out_path, seeds=(1, 2, 3), hidden=(48, 24)):
    Xtr, ytr = load_split(os.path.join(root, 'train')); Xva, yva = load_split(os.path.join(root, 'valid'))
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; A = (Xtr - mu) / sd; B = (Xva - mu) / sd
    best = None; log = []
    for s in seeds:
        net = MLP([NFEAT, *hidden, 7], seed=s); net.fit(A, ytr, seed=s)
        f1, cm = macro_f1(yva, net.predict(B)); ftr, _ = macro_f1(ytr, net.predict(A)); log.append((s, ftr, f1))
        if best is None or f1 > best[0]: best = (f1, net, cm, s)
    json.dump(dict(mu=mu.tolist(), sd=sd.tolist(), net=best[1].to_json(), valid_macro_f1=best[0], seed=best[3], seeds=log), open(out_path, 'w'))
    return best[0], best[2], log
