"""Record a model's cross-validation score in logs/cv_scores.csv, always on the same 5 prompt folds of train.csv.

Scores are balanced accuracy (the competition metric) per fold, so two rows can be compared fold by fold. The folds only
define how rows are grouped for scoring: 'kind' says how the predictions were made, and only out-of-fold predictions from
models that trained on train.csv are honest for those models (see the kinds below). A name already in the file is replaced.

    python src/cv_track.py oof_mix_Qwen3-4B-Base-5fold_full.npy --name Qwen3-4B-mix --kind 5-fold-oof --note "pool + train.csv x5, lr 3e-5"

Kinds: pool-only (never saw train.csv), 5-fold-oof, 2-fold-oof (trained on half of train.csv: understates the deployed
model), blend-in-sample (weights chosen on train.csv: optimistic).
"""
import argparse, os, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

OUT = 'logs/cv_scores.csv'; KINDS = ('pool-only', '5-fold-oof', '2-fold-oof', 'blend-in-sample')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npy'); ap.add_argument('--name', required=True); ap.add_argument('--kind', required=True, choices=KINDS); ap.add_argument('--note', default='')
    a = ap.parse_args()
    t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); y = t.model.map({l: i for i, l in enumerate(labels)}).values
    P = np.load(a.npy); assert P.shape == (len(t), len(labels)), P.shape
    assert (P.sum(1) > 0).all(), 'some rows have no prediction'
    pred = P.argmax(1); fold = [balanced_accuracy_score(y[te], pred[te]) for _, te in GroupKFold(5).split(t, y, t.prompt_id)]
    row = dict(date=time.strftime('%Y-%m-%d'), name=a.name, kind=a.kind, **{f'fold{i}': round(s, 4) for i, s in enumerate(fold)},
               mean=round(float(np.mean(fold)), 4), std=round(float(np.std(fold)), 4), file=a.npy, note=a.note)
    df = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=list(row))
    df = pd.concat([df[df.name != a.name], pd.DataFrame([row])], ignore_index=True).sort_values('mean', ascending=False)
    df.to_csv(OUT, index=False)
    print(f"{a.name:28s} {a.kind:16s} mean {row['mean']:.4f} +/- {row['std']:.4f} | " + ' '.join(f'{s:.3f}' for s in fold), flush=True)


if __name__ == '__main__':
    main()
