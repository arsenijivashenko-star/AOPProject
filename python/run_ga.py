import sys, json, time, statistics as st
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ga, classify
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = [2001, 2002, 2003]
import tempfile, shutil, csv
ROOT = os.environ.get('SARC_PROJECT', os.path.dirname(HERE)); BENCH = os.path.join(tempfile.gettempdir(), 'sarc_bench')
if not os.path.isdir(BENCH):
    os.makedirs(BENCH)
    for r in csv.DictReader(open(ROOT + '/corpus/labels.csv', encoding='utf-8')): shutil.copy(ROOT + '/corpus/' + r['file'], BENCH)
clf = classify.Classifier(os.path.join(HERE, 'model.json'))
res = {}
for name in ['ga', 'random', 'local']:
    runs = []
    for s in SEEDS:
        ev = ga.Evaluator(BENCH, clf); t = time.time()
        if name == 'ga': best, q, hist = ga.run_ga(ev, BUDGET, seed=s)
        elif name == 'random': best, q, hist = ga.run_random(ev, BUDGET, seed=s)
        else: best, q, hist = ga.run_local(ev, BUDGET, seed=s)
        runs.append(dict(seed=s, best=best, Q=q, hist=hist, evals=ev.n, seconds=time.time() - t, points=list(ev.points)))
        print(name, s, 'Q=%.4f' % q, 'evals', ev.n, best, flush=True)
    res[name] = runs
    print(name, 'mean Q %.4f  sd %.4f' % (st.mean(r['Q'] for r in runs), st.pstdev(r['Q'] for r in runs)), flush=True)
allp = [p for v in res.values() for r in v for p in r['points']]
res['pareto'] = ga.pareto(allp)
json.dump(res, open(os.path.join(HERE, 'ga_log.json'), 'w'), indent=1, default=str)
print('DONE pareto size', len(res['pareto']))
