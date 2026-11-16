#!/usr/bin/env python3
"""SARC smart secure archiver - command line.
  sarc_cli.py archive INPUT OUTPUT_BASE -p PASSWORD [--max-volume N] [--profile fast|balanced|best | --strategy greedy|lazy|dp|dp-iter]
                      [--recursive] [--dry-run] [--progress] [--json]
  sarc_cli.py extract ARCHIVE OUTDIR -p PASSWORD [--only NAME ...]   (ARCHIVE: base name, a .arc file, or several volume files)
  password: -p PASSWORD | --password-file FILE ('-' = stdin) | $SARC_PASSWORD | interactive prompt
  --json prints one machine-readable JSON object to stdout (errors too); --progress writes a progress bar to stderr
  sarc_cli.py list ARCHIVE -p PASSWORD                  (file table: names, sizes, CRC32, type, strategy, volumes)
  sarc_cli.py verify ARCHIVE -p PASSWORD                (full integrity check, writes nothing)
  sarc_cli.py info VOLUME.arc                           (public header only; nothing is decrypted)
  sarc_cli.py train DATA_ROOT                           (train the perceptron classifier on train/ + valid/)
  sarc_cli.py attack weak|integrity ...
"""
import sys, os, argparse, logging, getpass, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import SarcError, parse_header
import archive, codec

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'model.json')
STRATS = {'greedy': codec.GREEDY, 'lazy': codec.LAZY, 'dp': codec.DP, 'dp-iter': codec.DP2}

def _pw(a):
    if a.password is not None: return a.password
    pf = getattr(a, 'password_file', None)
    if pf:
        try:
            if pf == '-': raw = sys.stdin.readline()
            else:
                with open(pf, encoding='utf-8') as f: raw = f.readline()
        except (OSError, UnicodeDecodeError) as e: raise SarcError('E_INPUT', f'cannot read password file: {type(e).__name__}')
        pw = raw.rstrip('\r\n')
        if not pw: raise SarcError('E_INPUT', 'password file is empty')
        return pw
    env = os.environ.get('SARC_PASSWORD')
    return env if env else getpass.getpass('Password: ')

def _emit(a, obj, text=None):
    """--json: one JSON object on stdout; otherwise the human-readable text."""
    if getattr(a, 'json', False): print(json.dumps(obj, ensure_ascii=False))
    elif text is not None: print(text)

def _progress(a):
    if not getattr(a, 'progress', False): return None
    def cb(done, total):
        w = 30; k = w if total <= 0 else min(w, w * done // total)
        sys.stderr.write(f"\r[{'#' * k}{'.' * (w - k)}] {done}/{total} B"); sys.stderr.flush()
        if done >= total: sys.stderr.write('\n')
    return cb

def cmd_archive(a):
    clf = None
    if os.path.isfile(MODEL) and not a.no_classifier:
        import classify; clf = classify.Classifier(MODEL)
    if a.strategy:      # an explicit --strategy applies to every (non-stored) type; otherwise a profile decides per type
        params = codec.Params(strategy=STRATS[a.strategy], block_size=a.block_size); by_type = None
    else:
        by_type = archive.profile_strategies(a.profile); params = codec.Params(strategy=codec.LAZY, block_size=a.block_size)
    cfg = archive.Config(max_volume=a.max_volume, params=params, packer=a.packer, classifier=clf,
                         strategy_by_type=by_type, progress=_progress(a))
    if a.dry_run:
        st = archive.dry_run(a.input, cfg, a.recursive)
        ratio = st['orig_bytes'] / max(1, st['arc_bytes'])
        _emit(a, dict(dry_run=True, files=len(st['files']), orig_bytes=st['orig_bytes'], archive_bytes=st['arc_bytes'], ratio=round(ratio, 4),
                      volumes=st['volumes'], volume_bytes=st['volume_bytes'], blocks=st['blocks'], seconds=round(st['seconds'], 3),
                      plan=[dict(volume=i, W=v['W'], table=v['table'], blocks=v['n'], weight=v['weight'], value=v['value'])
                            for i, v in enumerate(st['vol_info'])]),
              f"dry run (nothing written): files={len(st['files'])} orig={st['orig_bytes']} predicted archive={st['arc_bytes']} "
              f"ratio={ratio:.3f} volumes={st['volumes']} blocks={st['blocks']} time={st['seconds']:.2f}s")
        if not a.json:
            for i, v in enumerate(st['vol_info']):
                print(f"  volume {i}: W={v['W']} B (file table {v['table']} B), {v['n']} block(s), knapsack weight {v['weight']}, value {v['value']}, size {st['volume_bytes'][i]} B")
        return 0
    st = {}
    outs = archive.create(a.input, a.output, _pw(a), cfg, a.recursive, st)
    ratio = st['orig_bytes'] / max(1, st['arc_bytes'])
    if a.json:
        _emit(a, dict(outputs=outs, files=len(st['files']), orig_bytes=st['orig_bytes'], archive_bytes=st['arc_bytes'],
                      ratio=round(ratio, 4), volumes=st['volumes'], seconds=round(st['seconds'], 3)))
        return 0
    for p in outs: print(f'wrote {p} ({os.path.getsize(p)} bytes)')
    print(f"files={len(st['files'])} orig={st['orig_bytes']} archive={st['arc_bytes']} ratio={ratio:.3f} volumes={st['volumes']} time={st['seconds']:.2f}s")
    return 0

def cmd_extract(a):
    arc = a.archive if len(a.archive) > 1 else a.archive[0]
    names = archive.extract(arc, a.outdir, _pw(a), a.only or None)
    _emit(a, dict(extracted=names, outdir=a.outdir, crc32_verified=True),
          f'extracted {len(names)} file(s) to {a.outdir}: all CRC32 verified' + (' (selected files only)' if a.only else '')); return 0

def cmd_list(a):
    arc = a.archive if len(a.archive) > 1 else a.archive[0]
    files, tot = archive.list_archive(arc, _pw(a))
    if a.json:
        _emit(a, dict(volumes=tot, files=[dict(name=f['name'], size=f['size'], crc32=f"{f['crc']:08x}", type=f['type'],
                      strategy=codec.STRAT_NAMES.get(f['strat'], f['strat']), volumes=f['volumes']) for f in files])); return 0
    print(f'{len(files)} file(s), {tot} volume(s)')
    for f in files:
        print(f"  {f['name']:<32} {f['size']:>9} B  crc32={f['crc']:08x}  type={f['type']}  strat={codec.STRAT_NAMES.get(f['strat'], f['strat'])}  vol={f['volumes']}")
    return 0

def cmd_verify(a):
    arc = a.archive if len(a.archive) > 1 else a.archive[0]
    n = archive.verify(arc, _pw(a))
    _emit(a, dict(ok=True, files_verified=n), f'OK: {n} file(s) verified (HMAC, structure, CRC32); nothing written'); return 0

def cmd_info(a):
    for p in a.volumes:
        d = open(p, 'rb').read()
        try: h = parse_header(d)
        except SarcError as e: print(f'{p}: INVALID ({e.code})'); continue
        print(f"{p}: SARC v1 volume {h['idx']}/{h['tot']} ciphertext={h['n']}B total={len(d)}B salt={h['salt'].hex()} (public)")
    return 0

def cmd_train(a):
    import classify
    f1, cm, log = classify.train(a.root, MODEL)
    print('seeds (seed, train F1, valid F1):', [(s, round(x, 4), round(y, 4)) for s, x, y in log])
    print('best valid macro-F1 = %.4f' % f1); print(cm); return 0

def cmd_attack(a):
    import attack
    if a.kind == 'integrity':
        d = open(a.archive, 'rb').read(); pw = _pw(a)
        res = attack.integrity_attack(d, pw, lambda b, p: archive.open_volume(b, p))
        bad = [r for r in res if r[1] == 'ACCEPTED']
        print(f'{len(res)} single-bit modifications tried; accepted by decoder: {len(bad)}')
        for p, c in res[:12]: print(f'  byte {p}: {c}')
        return 1 if bad else 0
    model = attack.build_model(os.path.join(a.data, 'train'))
    ct = open(a.archive, 'rb').read()
    fn = {'xor': attack.xor_dec, 'vigenere': attack.vig_dec}
    if a.cipher == 'subst':
        perm, f = attack.crack_substitution(ct, model); print('score', f); open(a.out, 'wb').write(attack.subst_dec(ct, perm)); return 0
    key, f, L = attack.crack_repeating_key(ct, model, fn[a.cipher])
    print(f'key length {L}, key {key.hex()}, score {f:.3f}'); open(a.out, 'wb').write(fn[a.cipher](ct, key)); return 0

def main(argv=None):
    ap = argparse.ArgumentParser(prog='sarc_cli', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--verbose', action='store_true'); ap.add_argument('--log', help='log file (never contains keys/passwords)')
    sp = ap.add_subparsers(dest='cmd', required=True)
    def common(p):
        p.add_argument('-p', '--password'); p.add_argument('--password-file', metavar='FILE', help="read the password from the first line of FILE ('-' = stdin)")
        p.add_argument('--json', action='store_true', help='machine-readable output'); p.add_argument('--progress', action='store_true', help='progress bar on stderr')
    p = sp.add_parser('archive'); p.add_argument('input'); p.add_argument('output')
    p.add_argument('--max-volume', type=int, default=65536); p.add_argument('--block-size', type=int, default=16384)
    p.add_argument('--strategy', choices=STRATS, help='one parser for every type (overrides --profile)')
    p.add_argument('--profile', choices=sorted(archive.PROFILES), default='balanced', help='fast = lazy; balanced = dp-iter on text/code/data, dp on raster/exe (default); best = dp-iter everywhere')
    p.add_argument('--packer', choices=['dp', 'greedy'], default='dp'); p.add_argument('--recursive', action='store_true')
    p.add_argument('--no-classifier', action='store_true', help='skip file-type detection (every file is treated as type 6)')
    p.add_argument('--dry-run', action='store_true', help='plan volumes and predict the exact archive size; writes nothing, no password needed')
    common(p); p.set_defaults(f=cmd_archive)
    p = sp.add_parser('extract'); p.add_argument('archive', nargs='+'); p.add_argument('outdir'); common(p)
    p.add_argument('--only', nargs='+', metavar='NAME', help='extract just these files (the rest are authenticated, not decoded)'); p.set_defaults(f=cmd_extract)
    p = sp.add_parser('list'); p.add_argument('archive', nargs='+'); common(p); p.set_defaults(f=cmd_list)
    p = sp.add_parser('verify'); p.add_argument('archive', nargs='+'); common(p); p.set_defaults(f=cmd_verify)
    p = sp.add_parser('info'); p.add_argument('volumes', nargs='+'); p.set_defaults(f=cmd_info)
    p = sp.add_parser('train'); p.add_argument('root'); p.set_defaults(f=cmd_train)
    p = sp.add_parser('attack'); p.add_argument('kind', choices=['weak', 'integrity']); p.add_argument('archive')
    p.add_argument('--cipher', choices=['xor', 'vigenere', 'subst'], default='xor'); p.add_argument('--data', default='.')
    p.add_argument('--out', default='recovered.bin'); p.add_argument('-p', '--password'); p.set_defaults(f=cmd_attack)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, filename=a.log, format='%(levelname)s %(message)s')
    try: return a.f(a)
    except SarcError as e: return _fail(a, e.code, str(e).split(': ', 1)[-1] if str(e).startswith(e.code + ': ') else str(e), 2)
    except BrokenPipeError:     # e.g. `sarc_cli list ... | head`: not an error, but stdout must not be flushed again
        try: sys.stdout = open(os.devnull, 'w')
        except OSError: pass
        return 0
    except KeyboardInterrupt: return _fail(a, 'E_INTERRUPTED', 'interrupted by user; nothing was left half-written', 130)
    except MemoryError: return _fail(a, 'E_MEMORY', 'out of memory; try a smaller --block-size', 3)
    except OSError as e: return _fail(a, 'E_IO', f'{type(e).__name__}: {e}', 3)

def _fail(a, code, msg, rc):
    if getattr(a, 'json', False): print(json.dumps(dict(error=code, message=msg), ensure_ascii=False))
    else: print(f'error {code}: {msg}', file=sys.stderr)
    return rc

if __name__ == '__main__': sys.exit(main())
