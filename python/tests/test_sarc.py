"""Acceptance tests C1 (12 cases) + edge cases C2 + security C5.  Run: python3 -m unittest -v test_sarc"""
import io, os, sys, json, glob, random, shutil, struct, tempfile, unittest, zlib, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import archive, codec, lz, huff, knapsack, fmt, attack
from fmt import SarcError, crc32
ROOT = os.environ.get('SARC_PROJECT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PW = 'algo2026'

def rb(path):
    with open(path, 'rb') as f: return f.read()
def wb(path, data):
    with open(path, 'wb') as f: f.write(data)
def rj(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def mk(files):
    d = tempfile.mkdtemp(prefix='sarc_t_')
    for n, b in files.items(): wb(os.path.join(d, n), b)
    return d

class Base(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(prefix='sarc_o_')
    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)
    def roundtrip(self, files, cfg=None, pw=PW):
        src = mk(files); st = {}
        outs = archive.create(src, self.tmp + '/a', pw, cfg, stats=st)
        archive.extract(self.tmp + '/a', self.tmp + '/o', pw)
        for n, b in files.items(): self.assertEqual(rb(self.tmp + '/o/' + n), b, n)
        shutil.rmtree(src); return outs, st

class IterativeDP(Base):
    """Expert level (case 19): extra passes never grow the block; round-trip stays byte-exact."""
    def test_iter_never_worse_than_pass0_and_roundtrips(self):
        for n in ['f0000.txt', 'f0001.py', 'f0002.csv', 'f0003.pgm']:
            d = rb(ROOT + '/corpus/' + n)[:8192]
            toks, hist = codec.parse_dp_iter(d, 4096, 32, 4, 3)
            self.assertEqual(lz.expand(toks), d)
            self.assertLessEqual(len(huff.encode_tokens(toks)), hist[0])
            self.assertLessEqual(len(codec.encode_block(d, codec.Params(strategy=codec.DP2))[1]), len(codec.encode_block(d, codec.Params(strategy=codec.DP))[1]))
    def test_iter_archive_roundtrip(self):
        files = {'a.txt': b'abracadabra ' * 300 + bytes(range(256)) * 4, 'b.dat': b'', 'c.bin': b'\x00' * 5000}
        self.roundtrip(files, archive.Config(params=codec.Params(strategy=codec.DP2)))

class SelectiveExtract(Base):
    """ROADMAP A: --only extracts chosen files; every volume is still authenticated."""
    FILES = {'a.txt': b'alpha beta gamma ' * 400, 'b.csv': b'1,2,3\n' * 900, 'c.bin': bytes(range(256)) * 20}
    def make(self, **kw):
        src = mk(self.FILES); archive.create(src, self.tmp + '/a', PW, archive.Config(**kw)); shutil.rmtree(src)
    def test_only_writes_just_requested_files(self):
        self.make()
        names = archive.extract(self.tmp + '/a', self.tmp + '/o', PW, only=['b.csv'])
        self.assertEqual(names, ['b.csv']); self.assertEqual(os.listdir(self.tmp + '/o'), ['b.csv'])
        self.assertEqual(rb(self.tmp + '/o/b.csv'), self.FILES['b.csv'])
    def test_only_unknown_name_is_controlled_error_and_writes_nothing(self):
        self.make()
        with self.assertRaises(SarcError) as c: archive.extract(self.tmp + '/a', self.tmp + '/o', PW, only=['nope.txt'])
        self.assertEqual(c.exception.code, 'E_INPUT'); self.assertFalse(os.path.exists(self.tmp + '/o'))
    def test_only_still_authenticates_whole_volume(self):
        self.make(); d = bytearray(rb(self.tmp + '/a.arc')); d[len(d) // 2] ^= 1; wb(self.tmp + '/a.arc', bytes(d))
        with self.assertRaises(SarcError) as c: archive.extract(self.tmp + '/a', self.tmp + '/o', PW, only=['a.txt'])
        self.assertEqual(c.exception.code, 'E_HMAC')
    def test_only_across_volumes(self):
        self.make(max_volume=4096, params=codec.Params(block_size=1024))
        self.assertGreater(len(glob.glob(self.tmp + '/a.part*.arc')), 1)
        archive.extract(self.tmp + '/a', self.tmp + '/o', PW, only=['a.txt', 'c.bin'])
        self.assertEqual(sorted(os.listdir(self.tmp + '/o')), ['a.txt', 'c.bin'])
        self.assertEqual(rb(self.tmp + '/o/c.bin'), self.FILES['c.bin'])

class DryRunAndProfiles(Base):
    def test_dry_run_predicts_exact_size_and_writes_nothing(self):
        src = mk({'a.txt': b'hello world ' * 3000, 'b.bin': os.urandom(20000)})
        cfg = archive.Config(max_volume=16384, params=codec.Params(block_size=4096))
        plan = archive.dry_run(src, cfg)
        self.assertEqual(os.listdir(self.tmp), [])
        outs = archive.create(src, self.tmp + '/a', PW, cfg)
        self.assertEqual(plan['volumes'], len(outs))
        self.assertEqual(plan['volume_bytes'], [os.path.getsize(p) for p in sorted(outs)])
        self.assertEqual(plan['arc_bytes'], sum(os.path.getsize(p) for p in outs))
    def test_dry_run_reports_capacity_error(self):
        src = mk({'a.txt': b'x' * 100})
        with self.assertRaises(SarcError) as c: archive.dry_run(src, archive.Config(max_volume=60))
        self.assertEqual(c.exception.code, 'E_CAPACITY')
    def test_profiles_roundtrip_and_strategy_recorded(self):
        files = {'t.txt': rb(ROOT + '/corpus/f0000.txt'), 'p.py': rb(ROOT + '/corpus/f0001.py')}
        sizes = {}
        for prof in archive.PROFILES:
            clf = None
            if os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.json')):
                import classify; clf = classify.Classifier(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.json'))
            cfg = archive.Config(strategy_by_type=archive.profile_strategies(prof), classifier=clf)
            src = mk(files); st = {}; outs = archive.create(src, self.tmp + '/' + prof, PW, cfg, stats=st)
            archive.extract(self.tmp + '/' + prof, self.tmp + '/o_' + prof, PW)
            for n, b in files.items(): self.assertEqual(rb(f'{self.tmp}/o_{prof}/{n}'), b)
            sizes[prof] = st['arc_bytes']
        self.assertLessEqual(sizes['balanced'], sizes['fast']); self.assertLessEqual(sizes['best'], sizes['fast'])
    def test_unknown_profile_rejected(self):
        with self.assertRaises(SarcError): archive.profile_strategies('turbo')
    def test_progress_callback_monotone_and_reaches_total(self):
        src = mk({'a.txt': b'abc ' * 9000, 'b.dat': b'z' * 5000}); seen = []
        archive.dry_run(src, archive.Config(params=codec.Params(block_size=4096), progress=lambda d, t: seen.append((d, t))))
        self.assertEqual(seen[-1][0], seen[-1][1]); self.assertEqual(seen, sorted(seen))

class CliFeatures(Base):
    def run_cli(self, *argv, stdin=None):
        import contextlib, sarc_cli
        out, err = io.StringIO(), io.StringIO(); old = sys.stdin
        if stdin is not None: sys.stdin = io.StringIO(stdin)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err): rc = sarc_cli.main(list(argv))
        finally: sys.stdin = old
        return rc, out.getvalue(), err.getvalue()
    def setUp(self):
        super().setUp(); self.src = mk({'a.txt': b'hello cli ' * 800, 'b.csv': b'1;2;3\n' * 500})
    def test_json_archive_list_verify_extract_with_password_file(self):
        pf = self.tmp + '/pw.txt'; wb(pf, PW.encode() + b'\n')
        rc, out, _ = self.run_cli('archive', self.src, self.tmp + '/a', '--password-file', pf, '--json')
        self.assertEqual(rc, 0); j = json.loads(out); self.assertEqual(j['files'], 2); self.assertGreater(j['ratio'], 1)
        rc, out, _ = self.run_cli('list', self.tmp + '/a', '--password-file', pf, '--json')
        self.assertEqual([f['name'] for f in json.loads(out)['files']], ['a.txt', 'b.csv'])
        rc, out, _ = self.run_cli('verify', self.tmp + '/a', '--password-file', pf, '--json'); self.assertEqual(json.loads(out), {'ok': True, 'files_verified': 2})
        rc, out, _ = self.run_cli('extract', self.tmp + '/a', self.tmp + '/o', '--only', 'b.csv', '--password-file', pf, '--json')
        self.assertEqual(json.loads(out)['extracted'], ['b.csv'])
    def test_password_from_stdin(self):
        rc, _, _ = self.run_cli('archive', self.src, self.tmp + '/a', '--password-file', '-', stdin=PW + '\n'); self.assertEqual(rc, 0)
        rc, _, _ = self.run_cli('verify', self.tmp + '/a', '-p', PW); self.assertEqual(rc, 0)
    def test_empty_or_missing_password_file_is_controlled_error(self):
        wb(self.tmp + '/e.txt', b'\n')
        for pf in (self.tmp + '/e.txt', self.tmp + '/missing.txt'):
            rc, out, err = self.run_cli('archive', self.src, self.tmp + '/a', '--password-file', pf)
            self.assertEqual(rc, 2); self.assertIn('E_INPUT', err); self.assertNotIn('Traceback', err)
    def test_dry_run_needs_no_password_and_writes_nothing(self):
        rc, out, _ = self.run_cli('archive', self.src, self.tmp + '/a', '--dry-run', '--json')
        self.assertEqual(rc, 0); j = json.loads(out); self.assertTrue(j['dry_run']); self.assertEqual(os.listdir(self.tmp), [])
        rc, out, _ = self.run_cli('archive', self.src, self.tmp + '/a', '-p', PW, '--json')
        self.assertEqual(json.loads(out)['archive_bytes'], j['archive_bytes'])
    def test_errors_as_json(self):
        rc, out, _ = self.run_cli('verify', self.tmp + '/none', '-p', PW, '--json')
        self.assertEqual(rc, 2); self.assertEqual(json.loads(out)['error'], 'E_INPUT')
    def test_wrong_password_exit_code_and_no_secret_in_output(self):
        self.run_cli('archive', self.src, self.tmp + '/a', '-p', PW)
        rc, out, err = self.run_cli('verify', self.tmp + '/a', '-p', 'wrong-password-123')
        self.assertEqual(rc, 2); self.assertNotIn('wrong-password-123', out + err)
    def test_explicit_strategy_overrides_profile(self):
        self.run_cli('archive', self.src, self.tmp + '/a', '-p', PW, '--strategy', 'greedy')
        rc, out, _ = self.run_cli('list', self.tmp + '/a', '-p', PW, '--json')
        self.assertEqual({f['strategy'] for f in json.loads(out)['files']} - {'store'}, {'greedy'})

class ListVerify(Base):
    def test_list_and_verify_multivolume(self):
        src = mk({'a.txt': b'hello ' * 5000, 'b.bin': os.urandom(30000)})
        outs = archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=16384, params=codec.Params(block_size=4096)))
        self.assertGreater(len(outs), 1)
        files, tot = archive.list_archive(self.tmp + '/a', PW)
        self.assertEqual((tot, [f['name'] for f in files]), (len(outs), ['a.txt', 'b.bin']))
        self.assertEqual(archive.verify(self.tmp + '/a', PW), 2)
        with self.assertRaises(SarcError) as c: archive.verify(self.tmp + '/a', 'wrong')
        self.assertEqual(c.exception.code, 'E_HMAC')
        with self.assertRaises(SarcError) as c: archive.list_archive(outs[1:], PW)
        self.assertEqual(c.exception.code, 'E_VOLUME_MISSING')
        self.assertFalse(os.path.exists(self.tmp + '/o'))      # verify/list never write extracted files
    def test_selective_extraction(self):
        src = mk({'keep.txt': b'keep me ' * 3000, 'skip.bin': os.urandom(24000), 'empty.dat': b''})
        outs = archive.create(src, self.tmp + '/selected', PW,
                              archive.Config(max_volume=16384, params=codec.Params(block_size=2048)))
        outdir = self.tmp + '/selected-out'
        names = archive.extract(self.tmp + '/selected', outdir, PW, only_names=['keep.txt', 'empty.dat'])
        self.assertEqual(names, ['empty.dat', 'keep.txt'])
        with open(outdir + '/keep.txt', 'rb') as f: self.assertEqual(f.read(), b'keep me ' * 3000)
        with open(outdir + '/empty.dat', 'rb') as f: self.assertEqual(f.read(), b'')
        self.assertFalse(os.path.exists(outdir + '/skip.bin'))
        with self.assertRaises(SarcError) as c:
            archive.extract(self.tmp + '/selected', self.tmp + '/missing-out', PW, only_names=['not-there.txt'])
        self.assertEqual(c.exception.code, 'E_INPUT')
        self.assertFalse(os.path.exists(self.tmp + '/missing-out'))
        with self.assertRaises(SarcError) as c:
            archive.extract(outs[1:], self.tmp + '/incomplete-out', PW, only_names=['keep.txt'])
        self.assertEqual(c.exception.code, 'E_VOLUME_MISSING')

class C1(Base):
    def test01_single_text(self): self.roundtrip({'a.txt': b'hello world, hello archive! ' * 40})
    def test02_empty_and_one_byte(self): self.roundtrip({'e.dat': b'', 'o.dat': b'x'})
    def test03_all_groups_corpus(self):
        corp = ROOT + '/corpus'; files = {n: rb(corp + '/' + n) for n in os.listdir(corp) if n != 'labels.csv'}
        _, st = self.roundtrip(files)
        self.assertEqual(st['orig_bytes'], 329903)
    def test04_identical_bytes_shrink_many_times(self):
        _, st = self.roundtrip({'z.dat': b'\x00' * 20000}); self.assertGreater(20000 / st['arc_bytes'], 5)
    def test05_noise_no_compression(self):
        r = random.Random(1); noise = bytes(r.randrange(256) for _ in range(10000))
        src = mk({'n.bin': noise}); st = {}; archive.create(src, self.tmp + '/a', PW, stats=st)
        fl, bl = fmt.parse_payload(fmt.open_volume(rb(self.tmp + '/a.arc'), PW)[1])
        self.assertTrue(all(b['flag'] == 1 and b['coded'] == b['raw'] for b in bl))
        self.assertEqual(st['arc_bytes'] - len(noise), fmt.HDR + fmt.TAG + 6 + 2 + len('n.bin') + 10 + 15)   # measured overhead
    def test06_already_compressed_stored(self):
        corp = ROOT + '/corpus'; d = rb(corp + '/f0005.z')
        src = mk({'f0005.z': d}); archive.create(src, self.tmp + '/a', PW)
        fl, bl = fmt.parse_payload(fmt.open_volume(rb(self.tmp + '/a.arc'), PW)[1])
        self.assertTrue(all(b['flag'] == 1 for b in bl))
    def test07_canonical_tree_from_lengths(self):
        lens = {65: 2, 66: 1, 67: 3, 68: 3}
        c = fmt.canon_codes(list(lens.items()))
        self.assertEqual(c[66], (0, 1)); self.assertEqual(c[65], (2, 2)); self.assertEqual(c[67], (6, 3)); self.assertEqual(c[68], (7, 3))
        enc = huff.tree_bytes(lens); got, _ = fmt.read_tree(enc, 0); self.assertEqual(dict(got), lens)
    def test08_degenerate_alphabet(self):
        _, st = self.roundtrip({'s.dat': b'A' * 5000}); self.assertLess(st['arc_bytes'], 200)
    def test09_greedy_and_lazy_sizes(self):
        d = rb(ROOT + '/corpus/f0001.py')[:16384]; sizes = {}
        for s, n in [(codec.GREEDY, 'greedy'), (codec.LAZY, 'lazy'), (codec.DP, 'dp')]:
            toks = {codec.GREEDY: lz.parse_greedy, codec.LAZY: lz.parse_lazy}.get(s, lambda x: lz.parse_dp(x)[0])(d)
            self.assertEqual(lz.expand(toks), d); sizes[n] = len(huff.encode_tokens(toks))
        self.assertLessEqual(sizes['lazy'], sizes['greedy']); self.assertLessEqual(sizes['dp'], sizes['lazy'])
    def test10_dp_knapsack_not_worse_than_greedy(self):
        r = random.Random(3)
        for _ in range(300):
            n = r.randint(1, 30); v = [r.randint(1, 999) for _ in range(n)]; w = [r.randint(1, 400) for _ in range(n)]; W = r.randint(50, 3000)
            self.assertGreaterEqual(knapsack.total(v, knapsack.knapsack_dp(v, w, W)), knapsack.total(v, knapsack.knapsack_greedy(v, w, W)))
    def test11_tamper_wrongpw_truncated(self):
        src = mk({'a.txt': b'secret data ' * 100}); archive.create(src, self.tmp + '/a', PW); d = rb(self.tmp + '/a.arc')
        b = bytearray(d); b[50] ^= 1
        with self.assertRaises(SarcError) as c: fmt.open_volume(bytes(b), PW)
        self.assertEqual(c.exception.code, 'E_HMAC')
        with self.assertRaises(SarcError) as c: fmt.open_volume(d, 'wrong')
        self.assertEqual(c.exception.code, 'E_HMAC')
        for cut in (0, 3, 40, 41, 72, len(d) - 1):
            with self.assertRaises(SarcError): fmt.open_volume(d[:cut], PW)
    def test12_volumes_order_missing_dup_big_file_reference(self):
        r = random.Random(9); big = bytes(r.randrange(256) for _ in range(70000)) + b'tail' * 3000
        src = mk({'big.bin': big, 'a.txt': b'abc ' * 3000}); st = {}
        outs = archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=32768), stats=st)
        self.assertGreater(len(outs), 1)
        sh = outs[::-1]; random.Random(1).shuffle(sh)
        names = archive.extract(sh, self.tmp + '/o', PW)                                  # any order
        self.assertEqual(rb(self.tmp + '/o/big.bin'), big)
        with self.assertRaises(SarcError) as c: archive.extract(outs[1:], self.tmp + '/o2', PW)
        self.assertEqual(c.exception.code, 'E_VOLUME_MISSING')
        with self.assertRaises(SarcError) as c: archive.extract(outs + [outs[0]], self.tmp + '/o3', PW)
        self.assertEqual(c.exception.code, 'E_VOLUME_DUP')
        self.assertFalse(os.path.exists(self.tmp + '/o2') and os.listdir(self.tmp + '/o2'))
        R = ROOT + '/reference'                                                           # teacher's reference archives
        for n in ['ref_01_single_file', 'ref_02_empty_file', 'ref_03_degenerate', 'ref_04_seven_groups', 'ref_05_no_compression', 'ref_06_two_volumes']:
            j = rj(f'{R}/{n}.json'); d = archive.decode_archive(archive.find_volumes(f'{R}/{n}'), PW)
            self.assertEqual({k: crc32(v) for k, v in d.items()}, {k: int(v['crc32'], 16) for k, v in j['вміст'].items()}, n)

class Reference(unittest.TestCase):
    def test_bad_archives_expected_codes(self):
        R = ROOT + '/reference'; exp = rj(R + '/expected_failures.json')
        for n, v in exp.items():
            with self.assertRaises(SarcError, msg=n) as c: archive.decode_archive([f'{R}/{n}'], PW)
            self.assertEqual(c.exception.code, v['очікуваний_код'], n)

class C2_C5(Base):
    def test_input_validation(self):
        with self.assertRaises(SarcError): archive.create(self.tmp + '/nope', self.tmp + '/a', PW)
        src = mk({}); 
        with self.assertRaises(SarcError): archive.create(src, self.tmp + '/a', PW)      # empty input set
        old = archive.MAX_NAME_BYTES; archive.MAX_NAME_BYTES = 10                         # file systems cap names at 255 B
        try:
            with self.assertRaises(SarcError) as c: archive.create(mk({'x' * 11: b'a'}), self.tmp + '/a', PW)      # name too long
            self.assertEqual(c.exception.code, 'E_INPUT')
        finally: archive.MAX_NAME_BYTES = old
        with self.assertRaises(SarcError): archive.create(mk({'a': b'1'}), self.tmp + '/a', '')
    def test_min_match_shorter_file(self): self.roundtrip({'t.txt': b'ab'})
    def test_tiny_volume_controlled_error(self):
        src = mk({'a.txt': b'hello ' * 50})
        with self.assertRaises(SarcError) as c: archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=60))
        self.assertEqual(c.exception.code, 'E_CAPACITY')
        with self.assertRaises(SarcError) as c: archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=100))
        self.assertEqual(c.exception.code, 'E_CAPACITY')
    def test_volume_ten_times_smaller(self):
        corp = ROOT + '/corpus'; files = {n: rb(corp + '/' + n) for n in ['f0000.txt', 'f0001.py', 'f0002.csv']}
        outs, st = self.roundtrip(files, archive.Config(max_volume=6554, params=codec.Params(block_size=4096)))
        self.assertGreaterEqual(len(outs), 3)
        for o in outs: self.assertLessEqual(os.path.getsize(o), 6554)
    def test_volume_size_limit_respected_default(self):
        corp = ROOT + '/corpus'; files = {n: rb(corp + '/' + n) for n in os.listdir(corp) if n != 'labels.csv'}
        outs, _ = self.roundtrip(files); [self.assertLessEqual(os.path.getsize(o), 65536) for o in outs]
    def test_payload_unsafe_names_rejected(self):
        for bad in ['../x', '/abs', 'a\\b', 'c:x', '..', '.', 'a\0b', '']:
            self.assertFalse(fmt.safe_name(bad), bad)
        # authenticated but structurally wrong payload -> E_PAYLOAD (never an unhandled exception)
        salt = b'1' * 16; ke, km = fmt.derive(PW.encode(), salt)
        for pt in [b'', b'\x00\x01\x00\x00', b'\xff' * 50, struct.pack('>HH', 1, 0) + struct.pack('>H', 3) + b'../' + struct.pack('>IIBB', 0, 0, 0, 0) + struct.pack('>H', 0)]:
            vol = archive.seal_volume(pt, ke, km, salt, b'n' * 8, 0, 1)
            with self.assertRaises(SarcError) as c: fmt.parse_payload(fmt.open_volume(vol, PW)[1])
            self.assertEqual(c.exception.code, 'E_PAYLOAD')
    def test_atomic_extraction_keeps_existing_files(self):
        src = mk({'a.txt': b'new content ' * 50}); archive.create(src, self.tmp + '/a', PW)
        os.makedirs(self.tmp + '/o'); wb(self.tmp + '/o/a.txt', b'OLD')
        with self.assertRaises(SarcError): archive.extract(self.tmp + '/a', self.tmp + '/o', 'wrongpw')
        self.assertEqual(rb(self.tmp + '/o/a.txt'), b'OLD')
        self.assertEqual([x for x in os.listdir(self.tmp + '/o')], ['a.txt'])
    def test_header_public_only_salt_nonce(self):
        src = mk({'secret_name.txt': b'x' * 500}); archive.create(src, self.tmp + '/a', PW); d = rb(self.tmp + '/a.arc')
        self.assertNotIn(b'secret_name', d); self.assertNotIn(PW.encode(), d)
    def test_fresh_salt_and_nonce_per_archive_distinct_per_volume(self):
        files = {'a.bin': bytes(random.Random(2).randrange(256) for _ in range(40000))}; src = mk(files)
        o1 = archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=16384))
        o2 = archive.create(src, self.tmp + '/b', PW, archive.Config(max_volume=16384))
        h1 = [fmt.parse_header(rb(p)) for p in o1]; h2 = [fmt.parse_header(rb(p)) for p in o2]
        self.assertEqual(len({h['salt'] for h in h1}), 1); self.assertNotEqual(h1[0]['salt'], h2[0]['salt'])
        self.assertEqual(len({h['nonce'] for h in h1}), len(h1))
    def test_integrity_every_bit_flip_rejected(self):
        src = mk({'a.txt': b'abc def ' * 200}); archive.create(src, self.tmp + '/a', PW); d = rb(self.tmp + '/a.arc')
        res = attack.integrity_attack(d, PW, lambda b, p: fmt.open_volume(b, p), n=120)
        self.assertFalse([r for r in res if r[1] == 'ACCEPTED'])
    def test_crc32_matches_zlib(self):
        r = random.Random(4)
        for n in (0, 1, 100, 5000): b = bytes(r.randrange(256) for _ in range(n)); self.assertEqual(crc32(b), zlib.crc32(b))
    def test_huffman_optimal_vs_bruteforce_small(self):
        # greedy Huffman cost == optimal prefix-code cost (checked against exhaustive search over length vectors)
        r = random.Random(8)
        for _ in range(60):
            k = r.randint(2, 5); f = {i: r.randint(1, 30) for i in range(k)}
            L = huff.code_lengths(f); cost = sum(f[s] * L[s] for s in f)
            best = min(sum(f[i] * l[i] for i in range(k)) for l in itertools.product(range(1, k + 1), repeat=k) if sum(2.0 ** -x for x in l) <= 1.0 + 1e-12)
            self.assertEqual(cost, best)
    def test_dp_parse_never_worse_and_exact_cost(self):
        r = random.Random(5)
        for _ in range(25):
            a = bytes(r.choice(b'abcab ') for _ in range(r.randint(1, 600)))
            g = lz.parse_greedy(a); toks, c = lz.parse_dp(a)
            self.assertEqual(lz.expand(toks), a); self.assertEqual(c, lz.tokens_cost(toks)); self.assertLessEqual(c, lz.tokens_cost(g))

class Classifier(unittest.TestCase):
    def test_valid_macro_f1_above_threshold(self):
        import classify
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model.json')): self.skipTest('no model')
        C = classify.Classifier(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model.json'))
        import csv
        rows = list(csv.DictReader(io.StringIO(rb(ROOT + '/valid/labels.csv').decode('utf-8'))))
        f1, _ = classify.macro_f1([int(r['type']) for r in rows], [C(rb(ROOT + '/valid/' + r['file'])) for r in rows])
        self.assertGreaterEqual(f1, 0.61)

class WeakCipherAttack(unittest.TestCase):
    def test_xor_and_vigenere_recovered(self):
        model = attack.build_model(ROOT + '/train')
        with open(ROOT + '/corpus/f0000.txt', 'rb') as f: pt = f.read()[:3000]
        key = b'\x5a\xc3\x17'
        ct = attack.xor_dec(pt, key)                                 # xor is its own inverse
        k, f, L = attack.crack_repeating_key(ct, model, attack.xor_dec, max_len=4, pop=40, gens=80)
        self.assertGreater(sum(a == b for a, b in zip(attack.xor_dec(ct, k), pt)) / len(pt), 0.95)

class ParallelEncoding(Base):
    """ROADMAP N1: --jobs N must not change a single coded byte."""
    FILES = {'a.txt': b'alpha beta gamma delta ' * 3000, 'b.csv': b'1,2,3\n' * 6000, 'c.bin': bytes(random.Random(1).randrange(256) for _ in range(20000)), 'd.dat': b''}
    def test_same_blocks_for_any_job_count(self):
        src = mk(self.FILES); paths = archive.collect(src)
        def run(j):
            files, cands = archive.analyze_and_encode(paths, archive.Config(params=codec.Params(block_size=4096), jobs=j))
            return files, [(c.file, c.idx, c.raw, c.flag, c.coded) for c in cands]
        one = run(1); two = run(2)
        self.assertEqual(one, two); self.assertGreater(len(one[1]), 8)
    def test_parallel_archive_roundtrip(self):
        self.roundtrip(self.FILES, archive.Config(params=codec.Params(block_size=4096), jobs=2))
    def test_jobs_below_one_is_clamped(self):
        self.assertEqual(archive.Config(jobs=0).jobs, 1); self.assertEqual(archive.Config(jobs=-3).jobs, 1)

class IncludeExclude(Base):
    """ROADMAP N4: glob filters for archive."""
    def tree(self):
        d = mk({'a.txt': b'a' * 100, 'b.tmp': b'b' * 100, 'c.py': b'c' * 100})
        os.makedirs(d + '/.git'); wb(d + '/.git/config', b'x' * 50); os.makedirs(d + '/sub'); wb(d + '/sub/e.txt', b'e' * 60)
        return d
    def names(self, d, **kw): return [os.path.basename(p) for p in archive.collect(d, **kw)]
    def test_exclude_glob_and_pruned_folder(self):
        d = self.tree()
        self.assertEqual(self.names(d, recursive=True, exclude=['*.tmp', '.git']), ['a.txt', 'c.py', 'e.txt'])
    def test_include_and_exclude_wins(self):
        d = self.tree()
        self.assertEqual(self.names(d, recursive=True, include=['*.txt', '*.py'], exclude=['a.*']), ['c.py', 'e.txt'])
    def test_relative_path_glob(self):
        d = self.tree(); self.assertEqual(self.names(d, recursive=True, include=['sub/*']), ['e.txt'])
    def test_filter_can_remove_duplicate_names(self):
        d = self.tree(); os.makedirs(d + '/other'); wb(d + '/other/e.txt', b'z')
        with self.assertRaises(SarcError): archive.collect(d, recursive=True)
        self.assertEqual(self.names(d, recursive=True, exclude=['other']), ['config', 'a.txt', 'b.tmp', 'c.py', 'e.txt'])  # sorted by full path: .git/config first
    def test_filter_everything_is_controlled_error(self):
        d = self.tree()
        with self.assertRaises(SarcError) as c: archive.create(d, self.tmp + '/a', PW, recursive=True, include=['*.nothing'])
        self.assertEqual(c.exception.code, 'E_INPUT')
    def test_archive_contains_only_selected(self):
        d = self.tree(); archive.create(d, self.tmp + '/a', PW, recursive=True, exclude=['*.tmp', '.git'])
        self.assertEqual([f['name'] for f in archive.list_archive(self.tmp + '/a', PW)[0]], ['a.txt', 'c.py', 'e.txt'])

class DiffAndBench(Base):
    """ROADMAP N3 (diff) and N2 (bench)."""
    FILES = {'a.txt': b'hello diff ' * 500, 'b.csv': b'1;2;3\n' * 400}
    def setUp(self):
        super().setUp(); self.src = mk(self.FILES); archive.create(self.src, self.tmp + '/a', PW)
    def test_identical_folder(self):
        r = archive.diff_archive(self.tmp + '/a', self.src, PW)
        self.assertEqual((r['same'], r['changed'], r['missing'], r['extra']), (['a.txt', 'b.csv'], [], [], []))
    def test_changed_missing_extra(self):
        b = bytearray(self.FILES['a.txt']); b[7] ^= 1; wb(self.src + '/a.txt', bytes(b)); os.remove(self.src + '/b.csv'); wb(self.src + '/new.txt', b'n')
        r = archive.diff_archive(self.tmp + '/a', self.src, PW)
        self.assertEqual((r['changed'], r['missing'], r['extra']), (['a.txt'], ['b.csv'], ['new.txt']))
    def test_wrong_password_and_not_a_folder(self):
        with self.assertRaises(SarcError) as c: archive.diff_archive(self.tmp + '/a', self.src, 'wrong')
        self.assertEqual(c.exception.code, 'E_HMAC')
        with self.assertRaises(SarcError) as c: archive.diff_archive(self.tmp + '/a', self.src + '/a.txt', PW)
        self.assertEqual(c.exception.code, 'E_INPUT')
    def test_cli_exit_codes(self):
        import contextlib, sarc_cli
        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out): rc = sarc_cli.main(list(argv))
            return rc, out.getvalue()
        rc, out = run('diff', self.tmp + '/a', self.src, '-p', PW, '--json'); self.assertEqual(rc, 0); self.assertTrue(json.loads(out)['identical'])
        wb(self.src + '/a.txt', b'changed'); rc, out = run('diff', self.tmp + '/a', self.src, '-p', PW)
        self.assertEqual(rc, 1); self.assertIn('~ a.txt', out)
    def test_bench_reports_q_and_roundtrip(self):
        import contextlib, sarc_cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out): rc = sarc_cli.main(['bench', self.src, '--profiles', 'fast', 'balanced', '--json'])
        j = json.loads(out.getvalue()); self.assertEqual(rc, 0); self.assertEqual([r['profile'] for r in j['results']], ['fast', 'balanced'])
        for r in j['results']:
            self.assertTrue(r['roundtrip_ok']); self.assertTrue(0 < r['Q'] < 1); self.assertAlmostEqual(r['Q'], 0.7 * r['S'] + 0.3 * r['T'], places=3)

class VolumeEdgeCases(Base):
    """ROADMAP #5: limits give controlled errors, never a traceback."""
    def test_volume_too_small_for_header(self):
        src = mk({'a.txt': b'x' * 100})
        with self.assertRaises(SarcError) as c: archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=41 + 32 + 6))
        self.assertEqual(c.exception.code, 'E_CAPACITY')
    def test_name_of_255_bytes_and_zero_size_roundtrip(self):
        self.roundtrip({'n' * 255: b'', 'z.txt': b'zero neighbour ' * 50})

if __name__ == '__main__': unittest.main()
