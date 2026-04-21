import numpy as np
from environment import likelihood_to_class

def compute_rmse(env, obs_likelihood):
    visited_mask = obs_likelihood >= 0
    if visited_mask.sum() == 0:
        return float('nan')
    diff = obs_likelihood[visited_mask] - env.likelihood_grid[visited_mask]
    return float(np.sqrt(np.mean(diff ** 2)))

def compute_confusion_matrix(env, obs_likelihood):
    visited_mask = obs_likelihood >= 0
    true_classes = np.array([likelihood_to_class(v) for v in env.likelihood_grid[visited_mask]])
    pred_classes = np.array([likelihood_to_class(v) for v in obs_likelihood[visited_mask]])
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        cm[t, p] += 1
    return cm, ['No Plant (0%)', 'Low (33%)', 'Medium (66%)', 'High (100%)']

def print_metrics(env, results):
    rmse  = results['rmse']
    cm    = results['confusion_matrix']
    names = results['class_names']
    print(f"\n=== Evaluation Metrics ===")
    print(f"Coverage      : {results['coverage_percent']:.1f}%  "
          f"({results['n_plantable_found']} / {results['n_plantable_total']} plantable cells)")
    print(f"RMSE          : {rmse:.4f}")
    print(f"Compute time  : {results['computation_time']:.2f}s")
    print(f"\nConfusion Matrix:")
    header = f"{'':>15}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row_str = f"{row_name:>15}" + "".join(f"{cm[i,j]:>14}" for j in range(4))
        print(row_str)
    print(f"\nPer-class recall:")
    for i, name in enumerate(names):
        total   = cm[i].sum()
        correct = cm[i, i]
        pct     = (correct / total * 100) if total > 0 else 0.0
        print(f"  {name:>15}: {correct:>5} / {total:>5}  ({pct:.1f}%)")