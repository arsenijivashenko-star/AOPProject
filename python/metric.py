"""Unified quality metric Q = alpha*S + beta*T (team_config.json). Q = 0 if any file is not restored byte-exact."""
import os, time, tempfile, shutil, json
from archive import create, extract, Config
ALPHA, BETA, T_REF = 0.7, 0.3, 30.0

def measure(src_dir, cfg, password='algo2026', repeat=1):
    tmp = tempfile.mkdtemp(prefix='q_'); st = {}
    try:
        t0 = time.perf_counter(); create(src_dir, tmp + '/a', password, cfg, stats=st); tc = time.perf_counter() - t0
        t0 = time.perf_counter(); extract(tmp + '/a', tmp + '/o', password); td = time.perf_counter() - t0
        ok = all(os.path.isfile(tmp + '/o/' + n) and open(src_dir + '/' + n, 'rb').read() == open(tmp + '/o/' + n, 'rb').read()
                 for n in os.listdir(src_dir) if os.path.isfile(src_dir + '/' + n))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    t = tc + td; S = st['orig_bytes'] / (st['orig_bytes'] + st['arc_bytes']); T = T_REF / (T_REF + t)
    Q = ALPHA * S + BETA * T if ok else 0.0
    return dict(Q=Q, S=S, T=T, t=t, t_comp=tc, t_dec=td, arc=st['arc_bytes'], orig=st['orig_bytes'], ok=ok, volumes=st['volumes'], stats=st)
