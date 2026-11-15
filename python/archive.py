"""SARC archiver: create (analysis -> parse -> huffman -> knapsack packing -> AES-CTR + HMAC) and extract (atomic)."""
import os, struct, time, secrets, tempfile, shutil, logging, glob, hmac, hashlib
from fmt import *
from codec import *
import knapsack

log = logging.getLogger('sarc')
TYPE_PRIORITY = {0: 3, 1: 3, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}      # team_config.json
MAX_NAME_BYTES = 255

class Config:
    def __init__(self, max_volume=65536, params=None, packer='dp', classifier=None, strategy_by_type=None):
        self.max_volume = max_volume; self.params = params or Params(); self.packer = packer
        self.classifier = classifier
        # per-type strategy chosen by the analysis block; type 5 (already compressed) -> store
        self.strategy_by_type = strategy_by_type or {5: STORE}

def _validate_input(path):
    if not os.path.isfile(path): raise SarcError('E_INPUT', f'not a file: {os.path.basename(path)}')
    if not os.access(path, os.R_OK): raise SarcError('E_INPUT', f'no read permission: {os.path.basename(path)}')
    name = os.path.basename(path)
    nb = name.encode('utf-8')
    if len(nb) > MAX_NAME_BYTES: raise SarcError('E_INPUT', 'file name too long')
    if not safe_name(name): raise SarcError('E_INPUT', f'unsafe file name: {name!r}')

def collect(input_path, recursive=False):
    if os.path.isfile(input_path): return [input_path]
    if not os.path.isdir(input_path): raise SarcError('E_INPUT', 'input does not exist')
    out = []
    if recursive:
        for r, _, fs in os.walk(input_path): out += [os.path.join(r, f) for f in fs]
    else:
        out = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    out.sort()
    names = [os.path.basename(p) for p in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup: raise SarcError('E_INPUT', f'duplicate file names (archive stores names only): {sorted(dup)[:3]}')
    return out

class Cand:     # candidate block
    __slots__ = ('file', 'idx', 'raw', 'flag', 'coded', 'value')

def analyze_and_encode(paths, cfg, stats=None):
    """Returns (files, cands). files: list of dict(name,size,crc,type,strat). Blocks are encoded BEFORE packing."""
    files = []; cands = []; bs = cfg.params.block_size
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
                flag, coded = encode_block(raw, p)
                c = Cand(); c.file = fi; c.idx = bi; c.raw = len(raw); c.flag = flag; c.coded = coded
                c.value = TYPE_PRIORITY.get(ftype, 1) * (len(raw) + 1)
                cands.append(c); bi += 1
                if not raw: break
        files.append(dict(name=os.path.basename(path), size=total, crc=crc, type=ftype, strat=strat))
        if stats is not None: stats.setdefault('per_file', []).append((os.path.basename(path), ftype, total))
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

def create(input_path, out_base, password, cfg=None, recursive=False, stats=None):
    """Returns list of written volume paths. Raises SarcError."""
    cfg = cfg or Config()
    if not password: raise SarcError('E_INPUT', 'empty password')
    t0 = time.time()
    paths = collect(input_path, recursive)
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

def decode_archive(vol_paths, password):
    """Returns {name: bytes}; verifies every volume (header -> HMAC -> payload -> CRC32). Nothing is written."""
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
    for name, m in metas.items():
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

def extract(archive, outdir, password):
    """Atomic: decode+verify everything first, write into a temp dir inside outdir, then move into place."""
    vols = find_volumes(archive)
    data = decode_archive(vols, password)
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
