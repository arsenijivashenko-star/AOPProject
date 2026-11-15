"""Acceptance tests C1 (12 cases) + edge cases C2 + security C5.  Run: python3 -m unittest -v test_sarc"""
import os, sys, json, glob, random, shutil, struct, tempfile, unittest, zlib, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import archive, codec, lz, huff, knapsack, fmt, attack
from fmt import SarcError, crc32
ROOT = os.environ.get('SARC_PROJECT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PW = 'algo2026'

def mk(files):
    d = tempfile.mkdtemp(prefix='sarc_t_')
    for n, b in files.items(): open(os.path.join(d, n), 'wb').write(b)
    return d

class Base(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(prefix='sarc_o_')
    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)
    def roundtrip(self, files, cfg=None, pw=PW):
        src = mk(files); st = {}
        outs = archive.create(src, self.tmp + '/a', pw, cfg, stats=st)
        archive.extract(self.tmp + '/a', self.tmp + '/o', pw)
        for n, b in files.items(): self.assertEqual(open(self.tmp + '/o/' + n, 'rb').read(), b, n)
        shutil.rmtree(src); return outs, st

class IterativeDP(Base):
    """Expert level (case 19): extra passes never grow the block; round-trip stays byte-exact."""
    def test_iter_never_worse_than_pass0_and_roundtrips(self):
        for n in ['f0000.txt', 'f0001.py', 'f0002.csv', 'f0003.pgm']:
            d = open(ROOT + '/corpus/' + n, 'rb').read()[:8192]
            toks, hist = codec.parse_dp_iter(d, 4096, 32, 4, 3)
            self.assertEqual(lz.expand(toks), d)
            self.assertLessEqual(len(huff.encode_tokens(toks)), hist[0])
            self.assertLessEqual(len(codec.encode_block(d, codec.Params(strategy=codec.DP2))[1]), len(codec.encode_block(d, codec.Params(strategy=codec.DP))[1]))
    def test_iter_archive_roundtrip(self):
        files = {'a.txt': b'abracadabra ' * 300 + bytes(range(256)) * 4, 'b.dat': b'', 'c.bin': b'\x00' * 5000}
        self.roundtrip(files, archive.Config(params=codec.Params(strategy=codec.DP2)))

class C1(Base):
    def test01_single_text(self): self.roundtrip({'a.txt': b'hello world, hello archive! ' * 40})
    def test02_empty_and_one_byte(self): self.roundtrip({'e.dat': b'', 'o.dat': b'x'})
    def test03_all_groups_corpus(self):
        corp = ROOT + '/corpus'; files = {n: open(corp + '/' + n, 'rb').read() for n in os.listdir(corp) if n != 'labels.csv'}
        _, st = self.roundtrip(files)
        self.assertEqual(st['orig_bytes'], 329903)
    def test04_identical_bytes_shrink_many_times(self):
        _, st = self.roundtrip({'z.dat': b'\x00' * 20000}); self.assertGreater(20000 / st['arc_bytes'], 5)
    def test05_noise_no_compression(self):
        r = random.Random(1); noise = bytes(r.randrange(256) for _ in range(10000))
        src = mk({'n.bin': noise}); st = {}; archive.create(src, self.tmp + '/a', PW, stats=st)
        fl, bl = fmt.parse_payload(fmt.open_volume(open(self.tmp + '/a.arc', 'rb').read(), PW)[1])
        self.assertTrue(all(b['flag'] == 1 and b['coded'] == b['raw'] for b in bl))
        self.assertEqual(st['arc_bytes'] - len(noise), fmt.HDR + fmt.TAG + 6 + 2 + len('n.bin') + 10 + 15)   # measured overhead
    def test06_already_compressed_stored(self):
        corp = ROOT + '/corpus'; d = open(corp + '/f0005.z', 'rb').read()
        src = mk({'f0005.z': d}); archive.create(src, self.tmp + '/a', PW)
        fl, bl = fmt.parse_payload(fmt.open_volume(open(self.tmp + '/a.arc', 'rb').read(), PW)[1])
        self.assertTrue(all(b['flag'] == 1 for b in bl))
    def test07_canonical_tree_from_lengths(self):
        lens = {65: 2, 66: 1, 67: 3, 68: 3}
        c = fmt.canon_codes(list(lens.items()))
        self.assertEqual(c[66], (0, 1)); self.assertEqual(c[65], (2, 2)); self.assertEqual(c[67], (6, 3)); self.assertEqual(c[68], (7, 3))
        enc = huff.tree_bytes(lens); got, _ = fmt.read_tree(enc, 0); self.assertEqual(dict(got), lens)
    def test08_degenerate_alphabet(self):
        _, st = self.roundtrip({'s.dat': b'A' * 5000}); self.assertLess(st['arc_bytes'], 200)
    def test09_greedy_and_lazy_sizes(self):
        d = open(ROOT + '/corpus/f0001.py', 'rb').read()[:16384]; sizes = {}
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
        src = mk({'a.txt': b'secret data ' * 100}); archive.create(src, self.tmp + '/a', PW); d = open(self.tmp + '/a.arc', 'rb').read()
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
        self.assertEqual(open(self.tmp + '/o/big.bin', 'rb').read(), big)
        with self.assertRaises(SarcError) as c: archive.extract(outs[1:], self.tmp + '/o2', PW)
        self.assertEqual(c.exception.code, 'E_VOLUME_MISSING')
        with self.assertRaises(SarcError) as c: archive.extract(outs + [outs[0]], self.tmp + '/o3', PW)
        self.assertEqual(c.exception.code, 'E_VOLUME_DUP')
        self.assertFalse(os.path.exists(self.tmp + '/o2') and os.listdir(self.tmp + '/o2'))
        R = ROOT + '/reference'                                                           # teacher's reference archives
        for n in ['ref_01_single_file', 'ref_02_empty_file', 'ref_03_degenerate', 'ref_04_seven_groups', 'ref_05_no_compression', 'ref_06_two_volumes']:
            j = json.load(open(f'{R}/{n}.json', encoding='utf-8')); d = archive.decode_archive(archive.find_volumes(f'{R}/{n}'), PW)
            self.assertEqual({k: crc32(v) for k, v in d.items()}, {k: int(v['crc32'], 16) for k, v in j['вміст'].items()}, n)

class Reference(unittest.TestCase):
    def test_bad_archives_expected_codes(self):
        R = ROOT + '/reference'; exp = json.load(open(R + '/expected_failures.json', encoding='utf-8'))
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
        corp = ROOT + '/corpus'; files = {n: open(corp + '/' + n, 'rb').read() for n in ['f0000.txt', 'f0001.py', 'f0002.csv']}
        outs, st = self.roundtrip(files, archive.Config(max_volume=6554, params=codec.Params(block_size=4096)))
        self.assertGreaterEqual(len(outs), 3)
        for o in outs: self.assertLessEqual(os.path.getsize(o), 6554)
    def test_volume_size_limit_respected_default(self):
        corp = ROOT + '/corpus'; files = {n: open(corp + '/' + n, 'rb').read() for n in os.listdir(corp) if n != 'labels.csv'}
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
        os.makedirs(self.tmp + '/o'); open(self.tmp + '/o/a.txt', 'wb').write(b'OLD')
        with self.assertRaises(SarcError): archive.extract(self.tmp + '/a', self.tmp + '/o', 'wrongpw')
        self.assertEqual(open(self.tmp + '/o/a.txt', 'rb').read(), b'OLD')
        self.assertEqual([x for x in os.listdir(self.tmp + '/o')], ['a.txt'])
    def test_header_public_only_salt_nonce(self):
        src = mk({'secret_name.txt': b'x' * 500}); archive.create(src, self.tmp + '/a', PW); d = open(self.tmp + '/a.arc', 'rb').read()
        self.assertNotIn(b'secret_name', d); self.assertNotIn(PW.encode(), d)
    def test_fresh_salt_and_nonce_per_archive_distinct_per_volume(self):
        files = {'a.bin': bytes(random.Random(2).randrange(256) for _ in range(40000))}; src = mk(files)
        o1 = archive.create(src, self.tmp + '/a', PW, archive.Config(max_volume=16384))
        o2 = archive.create(src, self.tmp + '/b', PW, archive.Config(max_volume=16384))
        h1 = [fmt.parse_header(open(p, 'rb').read()) for p in o1]; h2 = [fmt.parse_header(open(p, 'rb').read()) for p in o2]
        self.assertEqual(len({h['salt'] for h in h1}), 1); self.assertNotEqual(h1[0]['salt'], h2[0]['salt'])
        self.assertEqual(len({h['nonce'] for h in h1}), len(h1))
    def test_integrity_every_bit_flip_rejected(self):
        src = mk({'a.txt': b'abc def ' * 200}); archive.create(src, self.tmp + '/a', PW); d = open(self.tmp + '/a.arc', 'rb').read()
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
        rows = list(csv.DictReader(open(ROOT + '/valid/labels.csv', encoding='utf-8')))
        f1, _ = classify.macro_f1([int(r['type']) for r in rows], [C(open(ROOT + '/valid/' + r['file'], 'rb').read()) for r in rows])
        self.assertGreaterEqual(f1, 0.61)

class WeakCipherAttack(unittest.TestCase):
    def test_xor_and_vigenere_recovered(self):
        model = attack.build_model(ROOT + '/train')
        with open(ROOT + '/corpus/f0000.txt', 'rb') as f: pt = f.read()[:3000]
        key = b'\x5a\xc3\x17'
        ct = attack.xor_dec(pt, key)                                 # xor is its own inverse
        k, f, L = attack.crack_repeating_key(ct, model, attack.xor_dec, max_len=4, pop=40, gens=80)
        self.assertGreater(sum(a == b for a, b in zip(attack.xor_dec(ct, k), pt)) / len(pt), 0.95)

if __name__ == '__main__': unittest.main()
