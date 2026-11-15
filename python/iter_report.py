"""Case 19 (expert): iterative DP parsing. For every corpus file (blocks of 16384 B) the real coded size after pass 0
(fixed-cost DP), pass 1, 2, 3, 4 (costs = Huffman lengths of the previous pass). Prints a markdown table (REPORT_TABLES.md §7)."""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec, lz
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NAMES = ['природний текст', 'вихідний код', 'структуровані дані', 'растр без стиснення', 'виконуваний файл', 'вже стиснуте', 'штучні дані']
ITERS = 4
labels = {r['file']: int(r['type']) for r in csv.DictReader(open(os.path.join(ROOT, 'corpus', 'labels.csv')))}
tot = {t: [0] * (ITERS + 1) for t in range(7)}; lazy = {t: 0 for t in range(7)}; sec = 0.0
for n, t in sorted(labels.items()):
    d = open(os.path.join(ROOT, 'corpus', n), 'rb').read()
    for s in range(0, max(1, len(d)), 16384):
        raw = d[s:s + 16384]
        if not raw: continue
        t0 = time.perf_counter(); toks, h = codec.parse_dp_iter(raw, 4096, 32, 4, ITERS); sec += time.perf_counter() - t0
        assert lz.expand(toks) == raw
        for i, x in enumerate(h): tot[t][i] += min(x, len(raw))        # STORE fallback caps a block at raw size
        lazy[t] += len(codec.encode_block(raw, codec.Params(strategy=codec.LAZY))[1])
print('| група | лінивий | прохід 0 (фікс. вартості) | прохід 1 | прохід 2 | прохід 3 | прохід 4 | виграш 0→4 |')
print('|---|---|---|---|---|---|---|---|')
for t in range(7):
    h = tot[t]; print(f'| {NAMES[t]} | {lazy[t]} | ' + ' | '.join(map(str, h)) + f' | {100 * (h[0] - h[-1]) / max(1, h[0]):.2f}% |')
H = [sum(tot[t][i] for t in range(7)) for i in range(ITERS + 1)]; L = sum(lazy.values())
print(f'| **разом** | **{L}** | ' + ' | '.join(f'**{x}**' for x in H) + f' | **{100 * (H[0] - H[-1]) / H[0]:.2f}%** |')
print('\nПриріст за прохід (байти): ' + ', '.join(f'{H[i] - H[i + 1]}' for i in range(ITERS)) + f'; час {sec:.1f} с')
