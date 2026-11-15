#!/usr/bin/env python3
"""SARC smart secure archiver - command line.
  sarc_cli.py archive INPUT OUTPUT_BASE -p PASSWORD [--max-volume N] [--strategy greedy|lazy|dp|dp-iter] [--recursive]
  sarc_cli.py extract ARCHIVE OUTDIR -p PASSWORD        (ARCHIVE: base name, a .arc file, or several volume files)
  sarc_cli.py info VOLUME.arc                           (public header only; nothing is decrypted)
  sarc_cli.py train DATA_ROOT                           (train the perceptron classifier on train/ + valid/)
  sarc_cli.py attack weak|integrity ...
"""
import sys, os, argparse, logging, getpass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import SarcError, parse_header
import archive, codec

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'model.json')
STRATS = {'greedy': codec.GREEDY, 'lazy': codec.LAZY, 'dp': codec.DP, 'dp-iter': codec.DP2}

def _pw(a):
    if a.password is not None: return a.password
    env = os.environ.get('SARC_PASSWORD')
    return env if env else getpass.getpass('Password: ')

def cmd_archive(a):
    clf = None
    if os.path.isfile(MODEL) and not a.no_classifier:
        import classify; clf = classify.Classifier(MODEL)
    cfg = archive.Config(max_volume=a.max_volume, params=codec.Params(strategy=STRATS[a.strategy], block_size=a.block_size),
                         packer=a.packer, classifier=clf)
    st = {}
    outs = archive.create(a.input, a.output, _pw(a), cfg, a.recursive, st)
    for p in outs: print(f'wrote {p} ({os.path.getsize(p)} bytes)')
    print(f"files={len(st['files'])} orig={st['orig_bytes']} archive={st['arc_bytes']} "
          f"ratio={st['orig_bytes'] / max(1, st['arc_bytes']):.3f} volumes={st['volumes']} time={st['seconds']:.2f}s")
    return 0

def cmd_extract(a):
    arc = a.archive if len(a.archive) > 1 else a.archive[0]
    names = archive.extract(arc, a.outdir, _pw(a))
    print(f'extracted {len(names)} file(s) to {a.outdir}: all CRC32 verified'); return 0

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
    p = sp.add_parser('archive'); p.add_argument('input'); p.add_argument('output'); p.add_argument('-p', '--password')
    p.add_argument('--max-volume', type=int, default=65536); p.add_argument('--block-size', type=int, default=16384)
    p.add_argument('--strategy', choices=STRATS, default='lazy'); p.add_argument('--packer', choices=['dp', 'greedy'], default='dp')
    p.add_argument('--recursive', action='store_true'); p.add_argument('--no-classifier', action='store_true'); p.set_defaults(f=cmd_archive)
    p = sp.add_parser('extract'); p.add_argument('archive', nargs='+'); p.add_argument('outdir'); p.add_argument('-p', '--password'); p.set_defaults(f=cmd_extract)
    p = sp.add_parser('info'); p.add_argument('volumes', nargs='+'); p.set_defaults(f=cmd_info)
    p = sp.add_parser('train'); p.add_argument('root'); p.set_defaults(f=cmd_train)
    p = sp.add_parser('attack'); p.add_argument('kind', choices=['weak', 'integrity']); p.add_argument('archive')
    p.add_argument('--cipher', choices=['xor', 'vigenere', 'subst'], default='xor'); p.add_argument('--data', default='.')
    p.add_argument('--out', default='recovered.bin'); p.add_argument('-p', '--password'); p.set_defaults(f=cmd_attack)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, filename=a.log, format='%(levelname)s %(message)s')
    try: return a.f(a)
    except SarcError as e:
        print(f'error {e.code}: {e}', file=sys.stderr); return 2
    except (OSError, MemoryError) as e:
        print(f'error E_IO: {type(e).__name__}: {e}', file=sys.stderr); return 3

if __name__ == '__main__': sys.exit(main())
