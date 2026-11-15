"""Generates REPORT.md tables (spec section 12) from real measurements. Run on an idle machine (timings)."""
import os, sys, csv, json, time, shutil, statistics as st, tempfile, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import archive, codec, lz, huff, knapsack, metric, classify, fmt, secrets
from codec import Params, GREEDY, LAZY, DP, STORE

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('SARC_PROJECT', os.path.dirname(HERE))
CORP = ROOT + '/corpus'; BENCH = os.path.join(__import__('tempfile').gettempdir(), 'sarc_bench')
GROUPS = classify.CLASSES
clf = classify.Classifier(HERE + '/model.json')
rows = list(csv.DictReader(open(CORP + '/labels.csv', encoding='utf-8')))
if not os.path.isdir(BENCH):
    os.makedirs(BENCH); [shutil.copy(f"{CORP}/{r['file']}", BENCH) for r in rows]
out = []; P = out.append
def table(head, body):
    P('| ' + ' | '.join(head) + ' |'); P('|' + '---|' * len(head))
    for r in body: P('| ' + ' | '.join(str(x) for x in r) + ' |')
    P('')

def med_measure(cfg, rep=3):
    ms = [metric.measure(BENCH, cfg) for _ in range(rep)]
    ms.sort(key=lambda m: m['t']); m = ms[len(ms) // 2]; return m

# ---------------- T1: parse table by group
P('## 1. Розбір потоку: жадібний vs лінивий vs ДП (по групах корпусу)\n')
P('Розмір = байти після LZ77+Гаффман (блоки по 16384 Б, без шифрування), час = розбір+кодування (c).\n')
body = []
for k, g in enumerate(GROUPS):
    fs = [r['file'] for r in rows if int(r['type']) == k]; raw = sum(os.path.getsize(f'{CORP}/{f}') for f in fs); cell = {}
    for name, strat in [('greedy', GREEDY), ('lazy', LAZY), ('dp', DP)]:
        t = time.perf_counter(); size = 0
        for f in fs:
            d = open(f'{CORP}/{f}', 'rb').read()
            for s in range(0, max(1, len(d)), 16384): size += len(codec.encode_block(d[s:s + 16384], Params(strategy=strat))[1])
        cell[name] = (size, time.perf_counter() - t)
    body.append([g, raw] + [f'{cell[n][0]} / {cell[n][1]:.2f}' for n in ('greedy', 'lazy', 'dp')])
tot = [sum(int(r[i].split(' / ')[0]) for r in body) for i in (2, 3, 4)]
body.append(['**разом**', sum(r[1] for r in body)] + [f'**{x}**' for x in tot])
table(['група', 'сирий, Б', 'жадібний, Б / с', 'лінивий, Б / с', 'ДП, Б / с'], body)

# ---------------- T2: block impact on Q
P('## 2. Вплив кожного блоку на Q (корпус 21 файл, медіана з 3 запусків)\n')
base_clf = clf
variants = [
    ('повна система (лінивий, ДП-рюкзак, перцептрон)', archive.Config(params=Params(strategy=LAZY), classifier=clf)),
    ('без перцептрона (усі файли як тип 6, без відмови для типу 5)', archive.Config(params=Params(strategy=LAZY), classifier=None, strategy_by_type={})),
    ('жадібний замість лінивого', archive.Config(params=Params(strategy=GREEDY), classifier=clf)),
    ('ДП-розбір замість лінивого', archive.Config(params=Params(strategy=DP), classifier=clf)),
    ('жадібний рюкзак замість ДП-рюкзака', archive.Config(params=Params(strategy=LAZY), classifier=clf, packer='greedy')),
    ('без стиснення (усе STORE)', archive.Config(params=Params(strategy=STORE), classifier=clf, strategy_by_type={k: STORE for k in range(7)})),
    ('без відмови від стиснення (поріг ∞)', archive.Config(params=Params(strategy=LAZY, store_threshold=10.0), classifier=None, strategy_by_type={})),
]
body = []; res = {}
for name, cfg in variants:
    m = med_measure(cfg); res[name] = m
    body.append([name, m['arc'], f"{m['orig'] / m['arc']:.3f}", f"{m['t']:.2f}", f"{m['S']:.4f}", f"{m['T']:.4f}", f"**{m['Q']:.4f}**", m['volumes']])
table(['конфігурація', 'архів, Б', 'коеф.', 't, с', 'S', 'T', 'Q', 'томів'], body)
ref_arc = 128968
P(f'Опорний пакувальник викладача: {ref_arc} Б (коеф. {329903 / ref_arc:.3f}). Наш лінивий: {res[variants[0][0]]["arc"]} Б; ДП: {res[variants[3][0]]["arc"]} Б.\n')

# ---------------- T3: knapsack DP vs greedy
P('## 3. ДП проти жадібності на пакуванні першого тому\n')
body = []
for mv in (65536, 32768, 16384, 8192):
    cfg = archive.Config(max_volume=mv, params=Params(strategy=LAZY, block_size=min(16384, mv // 4)), classifier=clf)
    paths = archive.collect(BENCH); files, cands = archive.analyze_and_encode(paths, cfg)
    fset = sorted({c.file for c in cands}); table_b = sum(archive._table_bytes(files[i]) for i in fset); W = mv - 41 - 32 - 6 - table_b
    w = [len(c.coded) + 15 for c in cands]; v = [c.value for c in cands]
    t = time.perf_counter(); d = knapsack.knapsack_dp(v, w, W); td = time.perf_counter() - t
    t = time.perf_counter(); g = knapsack.knapsack_greedy(v, w, W); tg = time.perf_counter() - t
    vd, vg = knapsack.total(v, d), knapsack.total(v, g)
    body.append([mv, W, len(cands), vd, vg, vd - vg, f'{100 * (vd - vg) / vd:.3f}%', f'{td * 1000:.1f}', f'{tg * 1000:.2f}'])
table(['max_volume', 'W', 'блоків', 'цінність ДП', 'цінність жадібний', 'різниця', 'відн.', 'час ДП, мс', 'час жадібн., мс'], body)
P('Цінність блоку = пріоритет типу × (raw_len+1). ДП ≥ жадібного завжди (точний), жадібний вигравав у часі.\n')

# ---------------- T4: order of operations
P('## 4. Порядок операцій: стиснути→зашифрувати vs зашифрувати→стиснути\n')
ke, km = fmt.derive(b'algo2026', secrets.token_bytes(16)); nonce = secrets.token_bytes(12)
raw_tot = 0; ce = 0; ec = 0
for r in rows:
    d = open(f"{CORP}/{r['file']}", 'rb').read(); raw_tot += len(d)
    ce += sum(len(codec.encode_block(d[s:s + 16384], Params(strategy=LAZY))[1]) for s in range(0, max(1, len(d)), 16384))
    enc = fmt.ctr(ke, nonce, d)
    ec += sum(len(codec.encode_block(enc[s:s + 16384], Params(strategy=LAZY))[1]) for s in range(0, max(1, len(enc)), 16384))
table(['схема', 'розмір, Б', 'коеф. стиснення'], [['сирі дані', raw_tot, '1.000'], ['стиснути → зашифрувати (наша)', ce, f'{raw_tot / ce:.3f}'], ['зашифрувати → стиснути', ec, f'{raw_tot / ec:.3f}']])
P('AES-CTR дає шифртекст, статистично нерозрізнений від шуму (≈8 біт/байт); LZ77/Гаффман не знаходить повторів, тому блоки йдуть як STORE.\n')

# ---------------- T5: entropy by group
P('## 5. Ентропія по групах (чому «вже стиснуте» не стискається)\n')
body = []
for k, g in enumerate(GROUPS):
    fs = [r['file'] for r in rows if int(r['type']) == k]; H = []; rat = []
    for f in fs:
        d = open(f'{CORP}/{f}', 'rb').read(); c = np.bincount(np.frombuffer(d, dtype=np.uint8), minlength=256) / len(d); c = c[c > 0]
        H.append(float(-(c * np.log2(c)).sum())); rat.append(len(d) / sum(len(codec.encode_block(d[s:s + 16384], Params(strategy=LAZY))[1]) for s in range(0, len(d), 16384)))
    body.append([g, f'{st.mean(H):.3f}', f'{st.mean(rat):.2f}'])
table(['група', 'ентропія, біт/байт', 'коеф. стиснення'], body)

# ---------------- T6: classifier
P('## 6. Класифікатор (перцептрон з нуля: 61→48→24→7, ReLU, softmax, Adam)\n')
mj = json.load(open(HERE + '/model.json')); P(f"Ініціалізації (seed, F1 train, F1 valid): {[(s, round(a, 3), round(b, 3)) for s, a, b in mj['seeds']]}; обрано seed={mj['seed']} за macro-F1 на valid/.\n")
def conf(rows_, d):
    y = [int(r['type']) for r in rows_]; p = [clf(open(f"{d}/{r['file']}", 'rb').read()) for r in rows_]; return classify.macro_f1(y, p)
vrows = list(csv.DictReader(open(ROOT + '/valid/labels.csv', encoding='utf-8')))
for nm, rs, d in [('valid/', vrows, ROOT + '/valid'), ('corpus/ (не бачений при навчанні)', rows, CORP)]:
    f1, cm = conf(rs, d); P(f'**{nm}**: macro-F1 = {f1:.4f} (F1_min = 0.61; випадковий вибір ≈ 0.1614)\n')
    table(['факт\\прогноз'] + [str(i) for i in range(7)], [[str(i)] + [int(x) for x in cm[i]] for i in range(7)])

# ---------------- T7: GA
P('## 7. Збіжність ГА проти випадкового та локального пошуку (бюджет 40 обчислень пристосованості, 3 запуски)\n')
L = json.load(open(HERE + '/ga_log.json'))
rowsga = []
for nm in ('ga', 'random', 'local'):
    qs = [r['Q'] for r in L[nm]]; rowsga.append([nm, ' / '.join(f'{q:.4f}' for q in qs), f'{st.mean(qs):.4f}', f'{st.pstdev(qs):.4f}'])
table(['метод', 'Q за запусками', 'середнє', 'σ'], rowsga)
gh = [r['hist'] for r in L['ga']]
table(['запуск', 'найкраще gen 0', 'найкраще остання gen', 'монотонно'],
      [[r['seed'], f"{h[0]['best']:.4f}", f"{h[-1]['best']:.4f}", all(h[i]['best'] <= h[i + 1]['best'] + 1e-15 for i in range(len(h) - 1))] for r, h in zip(L['ga'], gh)])
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nm, c in [('ga', 'C0'), ('random', 'C1'), ('local', 'C2')]:
        for j, r in enumerate(L[nm]):
            h = r['hist']; xs = [x['evals'] for x in h]; ys = [x['best'] for x in h]; ax[0].step(xs, ys, c, alpha=.7, where='post', label=nm if j == 0 else None)
    ax[0].set_xlabel('обчислень пристосованості'); ax[0].set_ylabel('найкраще Q'); ax[0].legend(); ax[0].set_title('Збіжність (3 запуски кожного)')
    pf = L['pareto']; ax[1].scatter([p[0] for p in pf], [p[1] for p in pf], c='C3'); ax[1].set_xlabel('S (стиснення)'); ax[1].set_ylabel('T (швидкодія)'); ax[1].set_title('Фронт Парето (усі обчислені конфігурації)')
    plt.tight_layout(); plt.savefig(HERE + '/ga_convergence.png', dpi=110); P('![збіжність](ga_convergence.png)\n')
except Exception as e: P(f'(графік не побудовано: {e})\n')
P('Фронт Парето за (S, T):\n')
table(['S', 'T', 'стратегія', 'depth', 'min_match', 'block', 'thr'], [[f'{s:.4f}', f'{t:.4f}', i['strategy'], i['depth'], i['min_match'], i['block_size'], i['store_threshold']] for s, t, i in L['pareto']])
open(HERE + '/REPORT_TABLES.md', 'w', encoding='utf-8').write('\n'.join(out)); print('\n'.join(out))
