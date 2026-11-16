"""SARC archiver: create (analysis -> parse -> huffman -> knapsack packing -> AES-CTR + HMAC) and extract (atomic)."""
import os, struct, time, secrets, tempfile, shutil, logging, glob, hmac, hashlib, fnmatch
from concurrent.futures import ProcessPoolExecutor
from fmt import *
from codec import *
import knapsack

log = logging.getLogger('sarc')
TYPE_PRIORITY = {0: 3, 1: 3, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}      # team_config.json
MAX_NAME_BYTES = 255

class Config:
    def __init__(self, max_volume=65536, params=None, packer='dp', classifier=None, strategy_by_type=None, progress=None, jobs=1):
        self.max_volume = max_volume; self.params = params or Params(); self.packer = packer
        self.classifier = classifier
        self.jobs = max(1, int(jobs or 1))      # worker processes for block encoding; the archive is byte-identical for any value
        self.progress = progress        # optional callable(done_bytes, total_bytes), called after every encoded block
        # per-type strategy chosen by the analysis block; type 5 (already compressed) -> store
        self.strategy_by_type = strategy_by_type or {5: STORE}

# Profiles = which parser each detected file type gets (type ids: 0 text, 1 code, 2 structured, 3 raster, 4 exe, 5 already
# compressed, 6 artificial). Measured on corpus/ (REPORT_TABLES.md section 8): dp-iter gains 5-8% over lazy on text/code/
# structured data and 2-3% on raster/exe, while types 5 and 6 are incompressible, so spending time there is wasted.
PROFILES = {
    'fast':     {t: LAZY for t in range(7)} | {5: STORE},
    'balanced': {0: DP2, 1: DP2, 2: DP2, 3: DP, 4: DP, 5: STORE, 6: LAZY},
    'best':     {t: DP2 for t in range(7)} | {5: STORE},
}

def profile_strategies(name):
    if name not in PROFILES: raise SarcError('E_INPUT', f'unknown profile {name!r}; choose from {sorted(PROFILES)}')
    return dict(PROFILES[name])

def _validate_input(path):
    if not os.path.isfile(path): raise SarcError('E_INPUT', f'not a file: {os.path.basename(path)}')
    if not os.access(path, os.R_OK): raise SarcError('E_INPUT', f'no read permission: {os.path.basename(path)}')
    name = os.path.basename(path)
    nb = name.encode('utf-8')
    if len(nb) > MAX_NAME_BYTES: raise SarcError('E_INPUT', 'file name too long')
    if not safe_name(name): raise SarcError('E_INPUT', f'unsafe file name: {name!r}')

def _selected(rel, include, exclude):
    """Glob filter on the file name AND the relative path (so both '*.tmp' and 'build/*' work). exclude wins over include."""
    base = os.path.basename(rel); rel = rel.replace(os.sep, '/')
    hit = lambda pats: any(fnmatch.fnmatchcase(base, p) or fnmatch.fnmatchcase(rel, p) for p in pats)
    if exclude and hit(exclude): return False
    return not include or hit(include)

def collect(input_path, recursive=False, include=None, exclude=None):
    if os.path.isfile(input_path): return [input_path]
    if not os.path.isdir(input_path): raise SarcError('E_INPUT', 'input does not exist')
    out = []
    if recursive:
        for r, ds, fs in os.walk(input_path):
            ds[:] = sorted(d for d in ds if not (exclude and any(fnmatch.fnmatchcase(d, p) for p in exclude)))   # prune excluded folders
            out += [os.path.join(r, f) for f in fs]
    else:
        out = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    if include or exclude:
        out = [p for p in out if _selected(os.path.relpath(p, input_path), include, exclude)]
    out.sort()
    names = [os.path.basename(p) for p in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup: raise SarcError('E_INPUT', f'duplicate file names (archive stores names only): {sorted(dup)[:3]}')
    return out

class Cand:     # candidate block
    __slots__ = ('file', 'idx', 'raw', 'flag', 'coded', 'value')

def _enc(job):
    raw, p = job
    return encode_block(raw, p)

def analyze_and_encode(paths, cfg, stats=None):
    """Returns (files, cands). files: list of dict(name,size,crc,type,strat). Blocks are encoded BEFORE packing.
    cfg.jobs > 1 encodes blocks in worker processes. Blocks are independent and results are consumed in submission order,
    so the output is byte-identical to the sequential run (determinism is covered by a test)."""
    files = []; cands = []; bs = cfg.params.block_size
    total_bytes = sum(os.path.getsize(p) for p in paths if os.path.isfile(p)); done = 0
    batch = []; limit = 1 if cfg.jobs <= 1 else cfg.jobs * 4        # bounded memory: at most `limit` raw blocks in flight
    pool = ProcessPoolExecutor(max_workers=cfg.jobs) if cfg.jobs > 1 else None
    def flush():
        nonlocal done
        if not batch: return
        jobs = [(raw, p) for _, _, raw, p, _ in batch]
        res = list(pool.map(_enc, jobs)) if pool else [_enc(j) for j in jobs]
        for (fi, bi, raw, _, ftype), (flag, coded) in zip(batch, res):
            c = Cand(); c.file = fi; c.idx = bi; c.raw = len(raw); c.flag = flag; c.coded = coded
            c.value = TYPE_PRIORITY.get(ftype, 1) * (len(raw) + 1); cands.append(c); done += len(raw)
        batch.clear()
        if cfg.progress: cfg.progress(done, total_bytes)
    try:
        for fi, path in enumerate(paths):
            _validate_input(path)
            size = os.path.getsize(path)
            with open(path, 'rb') as f:
                head = f.read(min(size, 65536))                     # analysis sample (bounded memory)
                ftype = cfg.classifier(head) if cfg.classifier else 6
                strat = cfg.strategy_by_type.get(ftype, cfg.params.strategy)
                p = cfg.params.copy(strategy=strat)
                f.seek(0); crc = 0; bi = 0; total = 0
                while True:
                    raw = f.read(bs)
                    if not raw and bi > 0: break
                    crc = crc32(raw, crc); total += len(raw)
                    batch.append((fi, bi, raw, p, ftype)); bi += 1
                    if len(batch) >= limit: flush()
                    if not raw: break
            files.append(dict(name=os.path.basename(path), size=total, crc=crc, type=ftype, strat=strat))
            if stats is not None: stats.setdefault('per_file', []).append((os.path.basename(path), ftype, total))
        flush()
    finally:
        if pool: pool.shutdown(wait=True, cancel_futures=True)
    return files, cands

def _table_bytes(f): return 2 + len(f['name'].encode('utf-8')) + 10

def plan_volumes(files, cands, max_volume, packer='dp'):
    """Repeatedly: fix candidate files + reserve their table, W = max - 41 - 32 - 6 - table, solve 0/1 knapsack."""
    remaining = list(cands); plan = []; info = []
    while remaining:
        fset = sorted({c.file for c in remaining})
        table = sum(_table_bytes(files[i]) for i in fset)
        W = max_volume - HDR - TAG - 6 - table
        if W <= 0 or len(fset) > 65535:
            raise SarcError('E_CAPACITY', f'volume size {max_volume} too small for the file table ({table} bytes)')
        w = [len(c.coded) + 15 for c in remaining]; v = [c.value for c in remaining]
        pick = (knapsack.knapsack_dp if packer == 'dp' else knapsack.knapsack_greedy)(v, w, W)
        if not pick:
            raise SarcError('E_CAPACITY', f'no block fits: smallest weight {min(w)} > W={W}; reduce block_size or raise max_volume')
        if len(pick) > 65535: raise SarcError('E_CAPACITY', 'too many blocks in a volume')
        info.append(dict(W=W, table=table, value=knapsack.total(v, pick), weight=sum(w[i] for i in pick), n=len(pick)))
        ps = set(pick)
        plan.append((fset, [remaining[i] for i in pick]))
        remaining = [c for i, c in enumerate(remaining) if i not in ps]
    return plan, info

def build_payload(files, fset, blocks):
    pos = {fi: k for k, fi in enumerate(fset)}
    out = bytearray(struct.pack('>HH', len(fset), 0))
    for fi in fset:
        f = files[fi]; nb = f['name'].encode('utf-8')
        out += struct.pack('>H', len(nb)) + nb + struct.pack('>IIBB', f['size'], f['crc'], f['type'], f['strat'])
    blocks = sorted(blocks, key=lambda c: (c.file, c.idx))
    out += struct.pack('>H', len(blocks))
    for c in blocks: out += struct.pack('>HIIIB', pos[c.file], c.idx, c.raw, len(c.coded), c.flag)
    for c in blocks: out += c.coded
    return bytes(out)

def seal_volume(payload, ke, km, salt, nonce_prefix, idx, total):
    nonce = nonce_prefix + struct.pack('>I', idx)
    ct = ctr(ke, nonce, payload)
    head = b'SARC' + bytes([1]) + struct.pack('>HH', idx, total) + salt + nonce + struct.pack('>I', len(ct))
    assert len(head) == HDR
    return head + ct + hmac.new(km, head + ct, hashlib.sha256).digest()

def dry_run(input_path, cfg=None, recursive=False, include=None, exclude=None):
    """Plan an archive WITHOUT writing anything: encodes all blocks (weights must be exact), solves the knapsack for every
    volume and returns the exact size the archive would have. No password / key derivation is needed."""
    cfg = cfg or Config(); t0 = time.time()
    paths = collect(input_path, recursive, include, exclude)
    if not paths: raise SarcError('E_INPUT', 'no input files')
    if len(paths) > 65535: raise SarcError('E_INPUT', 'too many files')
    st = {}
    files, cands = analyze_and_encode(paths, cfg, st)
    plan, info = plan_volumes(files, cands, cfg.max_volume, cfg.packer)
    vols = [len(build_payload(files, fset, blocks)) + HDR + TAG for fset, blocks in plan]
    for i, v in enumerate(vols): info[i]['volume_bytes'] = v
    orig = sum(f['size'] for f in files)
    return dict(files=files, volumes=len(plan), vol_info=info, volume_bytes=vols, arc_bytes=sum(vols), orig_bytes=orig,
                blocks=len(cands), max_volume=cfg.max_volume, seconds=time.time() - t0)

def create(input_path, out_base, password, cfg=None, recursive=False, stats=None, include=None, exclude=None):
    """Returns list of written volume paths. Raises SarcError."""
    cfg = cfg or Config()
    if not password: raise SarcError('E_INPUT', 'empty password')
    t0 = time.time()
    paths = collect(input_path, recursive, include, exclude)
    if not paths: raise SarcError('E_INPUT', 'no input files')
    if len(paths) > 65535: raise SarcError('E_INPUT', 'too many files')
    files, cands = analyze_and_encode(paths, cfg, stats)
    plan, info = plan_volumes(files, cands, cfg.max_volume, cfg.packer)
    salt = secrets.token_bytes(16); nonce_prefix = secrets.token_bytes(8)       # fresh per archive
    ke, km = derive(password.encode('utf-8'), salt)                             # PBKDF2 once per archive
    vols = [seal_volume(build_payload(files, fset, blocks), ke, km, salt, nonce_prefix, i, len(plan))
            for i, (fset, blocks) in enumerate(plan)]
    outs = []
    d = os.path.dirname(os.path.abspath(out_base)); os.makedirs(d, exist_ok=True)
    for i, v in enumerate(vols):
        p = out_base + '.arc' if len(vols) == 1 else f'{out_base}.part{i}.arc'
        tmp = p + '.tmp'
        with open(tmp, 'wb') as f: f.write(v)
        os.replace(tmp, p); outs.append(p)
    if stats is not None:
        stats.update(volumes=len(vols), vol_info=info, arc_bytes=sum(len(v) for v in vols),
                     orig_bytes=sum(f['size'] for f in files), seconds=time.time() - t0,
                     files=files, blocks=len(cands),
                     coded_bytes=sum(len(c.coded) for c in cands))
    log.info('created %d volume(s) from %d files', len(vols), len(files))
    return outs

# ------------------------------- extraction -------------------------------
def find_volumes(arg):
    if isinstance(arg, (list, tuple)): return list(arg)
    if os.path.isfile(arg): return [arg]
    c = sorted(glob.glob(glob.escape(arg) + '.part*.arc'), key=lambda s: int(s.rsplit('.part', 1)[1][:-4]))
    if c: return c
    if os.path.isfile(arg + '.arc'): return [arg + '.arc']
    raise SarcError('E_INPUT', 'archive not found')

def decode_archive(vol_paths, password, only=None):
    """Returns {name: bytes}; verifies every volume (header -> HMAC -> payload -> CRC32). Nothing is written.
    only: optional collection of file names. Every volume is still authenticated (HMAC) and the whole file table is
    checked, but block data is decoded (and CRC32-checked) just for the requested files."""
    if not vol_paths: raise SarcError('E_INPUT', 'no volumes')
    cache = {}; seen = {}; tot = None; metas = {}; blocks = {}
    for p in vol_paths:
        with open(p, 'rb') as f: d = f.read()
        h, pt = open_volume(d, password, cache)
        if tot is None: tot = h['tot']
        elif h['tot'] != tot: raise SarcError('E_PAYLOAD', 'volumes belong to different sets')
        if h['idx'] in seen: raise SarcError('E_VOLUME_DUP', f"volume {h['idx']} supplied twice")
        seen[h['idx']] = p
        fl, bl = parse_payload(pt)
        for b in bl:
            f = fl[b['fi']]
            m = metas.setdefault(f['name'], dict(size=f['size'], crc=f['crc'], type=f['type']))
            if (m['size'], m['crc']) != (f['size'], f['crc']): raise SarcError('E_PAYLOAD', 'inconsistent file table')
            if b['off'] in blocks.setdefault(f['name'], {}): raise SarcError('E_PAYLOAD', 'duplicate block')
            blocks[f['name']][b['off']] = b
        for f in fl: metas.setdefault(f['name'], dict(size=f['size'], crc=f['crc'], type=f['type']))
    missing = [i for i in range(tot) if i not in seen]
    if missing: raise SarcError('E_VOLUME_MISSING', f'missing volume(s) {missing}')
    out = {}
    if only is not None:
        only = list(dict.fromkeys(only)); unknown = [n for n in only if n not in metas]
        if unknown: raise SarcError('E_INPUT', f'not in archive: {unknown[:3]}')
    for name, m in metas.items():
        if only is not None and name not in only: continue
        bl = blocks.get(name, {}); idxs = sorted(bl)
        if idxs != list(range(len(idxs))): raise SarcError('E_PAYLOAD', 'block gap')
        buf = bytearray()
        for i in idxs:
            b = bl[i]
            if b['flag'] == 1:
                if b['raw'] != b['coded']: raise SarcError('E_PAYLOAD', 'stored block length')
                buf += b['data']
            elif b['flag'] == 0: decode_block(b['data'], b['raw'], buf)
            else: raise SarcError('E_PAYLOAD', 'block flag')
            if len(buf) > m['size']: raise SarcError('E_PAYLOAD', 'size overflow')
        if len(buf) != m['size']: raise SarcError('E_PAYLOAD', 'size mismatch')
        if crc32(bytes(buf)) != m['crc']: raise SarcError('E_CRC', name)
        out[name] = bytes(buf)
    return out

def extract(archive, outdir, password, only=None):
    """Atomic: decode+verify everything first, write into a temp dir inside outdir, then move into place.
    only: optional list of file names to extract (the others are authenticated but not decoded)."""
    vols = find_volumes(archive)
    data = decode_archive(vols, password, only)
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='.sarc_', dir=outdir)
    try:
        for n, b in data.items():
            if not safe_name(n): raise SarcError('E_PAYLOAD', 'unsafe name')
            with open(os.path.join(tmp, n), 'wb') as f: f.write(b)
        for n in data: os.replace(os.path.join(tmp, n), os.path.join(outdir, n))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sorted(data)

def list_archive(arg, password):
    """File table of a (multi-volume) archive WITHOUT decoding block data: header -> HMAC -> payload structure checked.
    Returns (files, vol_count) with files = [{name,size,crc,type,strat,volumes:[...]}]."""
    vols = find_volumes(arg); cache = {}; seen = set(); tot = None; table = {}
    for p in vols:
        with open(p, 'rb') as f: d = f.read()
        h, pt = open_volume(d, password, cache)
        if tot is None: tot = h['tot']
        elif h['tot'] != tot: raise SarcError('E_PAYLOAD', 'volumes belong to different sets')
        if h['idx'] in seen: raise SarcError('E_VOLUME_DUP', f"volume {h['idx']} supplied twice")
        seen.add(h['idx'])
        fl, bl = parse_payload(pt)
        for f in fl: table.setdefault(f['name'], dict(f, volumes=[]))
        for b in bl:
            v = table[fl[b['fi']]['name']]['volumes']
            if h['idx'] not in v: v.append(h['idx'])
    missing = [i for i in range(tot) if i not in seen]
    if missing: raise SarcError('E_VOLUME_MISSING', f'missing volume(s) {missing}')
    return sorted(table.values(), key=lambda f: f['name']), tot

def verify(arg, password):
    """Full integrity check (HMAC, structure, decompression, size, CRC32 of every file) with nothing written to disk."""
    return len(decode_archive(find_volumes(arg), password))

def diff_archive(arg, directory, password, recursive=False):
    """Compare archive contents with a folder WITHOUT extracting to disk (ROADMAP N3). Returns
    dict(same=[...], changed=[...], missing=[...], extra=[...]): 'missing' = in the archive but not in the folder,
    'extra' = in the folder but not in the archive. The archive is fully authenticated and CRC-checked first."""
    if not os.path.isdir(directory): raise SarcError('E_INPUT', 'not a folder')
    data = decode_archive(find_volumes(arg), password)
    present = {}
    if recursive:
        for r, _, fs in os.walk(directory):
            for f in fs: present.setdefault(f, os.path.join(r, f))
    else:
        for f in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, f)): present[f] = os.path.join(directory, f)
    out = dict(same=[], changed=[], missing=[], extra=sorted(set(present) - set(data)))
    for n in sorted(data):
        if n not in present: out['missing'].append(n); continue
        with open(present[n], 'rb') as fh: cur = fh.read()
        out['same' if cur == data[n] else 'changed'].append(n)
    return out
