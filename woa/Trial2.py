import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
import time
import argparse
from src.whale_optimization import WhaleOptimization
from environment import Environment, COVERAGE_NONE, COVERAGE_PRESENT, COVERAGE_PATCHY, COVERAGE_CONTINUOUS
from metrics import compute_rmse, compute_confusion_matrix, print_metrics
from plot_results import plot_results

root = os.path.join(os.path.dirname(__file__), '..')

# ── Load precomputed grids (no need to rebuild environment) ───────────────
print("Loading environment...")
LonGrid    = np.load(os.path.join(root, 'lon_grid.npy'))
LatGrid    = np.load(os.path.join(root, 'lat_grid.npy'))
percover   = np.load(os.path.join(root, 'percover.npy'))
depth_grid = np.load(os.path.join(root, 'depth_grid.npy'))

coverage_grid = np.zeros(percover.shape, dtype=int)
coverage_grid[percover == 0.0]  = COVERAGE_CONTINUOUS
coverage_grid[percover == 0.25] = COVERAGE_PRESENT
coverage_grid[percover == 0.5]  = COVERAGE_PATCHY
coverage_grid[percover == 0.55] = COVERAGE_PRESENT
coverage_grid[percover >= 0.7]  = COVERAGE_NONE

env = Environment(depth_grid, coverage_grid)

# ── Planting function (unchanged from Trial2.py) ──────────────────────────
def planting(X, Y, LonGrid, LatGrid, seagrass_coverage, depth):
    y = np.argmin(np.abs(LatGrid[:,0] - Y))
    x = np.argmin(np.abs(LonGrid[0,:] - X))
    sc = seagrass_coverage[y, x]
    d  = depth[y, x]
    good_depth = (d > 1) and (d < 3)
    if good_depth and (sc == "51 - 100%"):    return 0.25
    if good_depth and (sc == "90 - 100%"):    return 0.5
    if good_depth and (sc == "1 - 89%"):      return 0.55
    if good_depth and (sc == "10 - 50%"):     return 0.7
    if good_depth and (sc == "Continuous"):   return 0
    if good_depth and (sc == "Discontinuous"): return 0.5
    if good_depth and (sc == "<50%"):         return 0.75
    if good_depth and (sc == "Unknown"):      return 0.5
    if good_depth and (sc == ">50%"):         return 0.25
    if good_depth and (sc == "Continuous Seagrass"): return 0
    if good_depth and (sc == "Patchy (Discontinuous) Seagrass"): return 0.5
    if d > 3:  return 0
    if d <= 1: return 0
    else:      return 0.5

# ── Convert WOA results to shared results dict ────────────────────────────
def build_results(env, all_solutions, LonGrid, LatGrid, elapsed):
    obs_likelihood = np.full((env.rows, env.cols), -1.0)
    obs_coverage   = np.full((env.rows, env.cols), -1, dtype=int)
    obs_depth      = np.full((env.rows, env.cols), np.nan)
    coverage_over_time = []
    trajectories = {}

    n_plantable = max(int(np.sum(env.likelihood_grid > 0)), 1)

    for gen_idx, generation in enumerate(all_solutions):
        for agent_idx, (fitness, (x, y)) in enumerate(generation):
            row = int(np.argmin(np.abs(LatGrid[:,0] - y)))
            col = int(np.argmin(np.abs(LonGrid[0,:] - x)))
            row = int(np.clip(row, 0, env.rows - 1))
            col = int(np.clip(col, 0, env.cols - 1))
            obs_likelihood[row, col] = env.sense_likelihood(row, col)
            obs_coverage[row, col]   = env.sense_coverage(row, col)
            obs_depth[row, col]      = env.sense_depth(row, col)
            if agent_idx not in trajectories:
                trajectories[agent_idx] = []
            trajectories[agent_idx].append((row, col))

        n_found = int(np.sum((obs_likelihood > 0) & (env.likelihood_grid > 0)))
        coverage_over_time.append(n_found / n_plantable * 100)

    n_found = int(np.sum((obs_likelihood > 0) & (env.likelihood_grid > 0)))
    cm, class_names = compute_confusion_matrix(env, obs_likelihood)

    return dict(
        observed_likelihood = obs_likelihood,
        observed_coverage   = obs_coverage,
        observed_depth      = obs_depth,
        coverage_percent    = n_found / n_plantable * 100,
        coverage_over_time  = coverage_over_time,
        n_plantable_found   = n_found,
        n_plantable_total   = n_plantable,
        rmse                = compute_rmse(env, obs_likelihood),
        confusion_matrix    = cm,
        class_names         = class_names,
        computation_time    = elapsed,
        trajectories        = list(trajectories.values()),
        fish_final_pos      = [(t[-1][0], t[-1][1]) for t in trajectories.values()],
    )

# ── Main ──────────────────────────────────────────────────────────────────
def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=100)
    parser.add_argument("-ngens", type=int, default=30)
    parser.add_argument("-a",     type=float, default=2.0)
    parser.add_argument("-b",     type=float, default=0.5)
    parser.add_argument("-max",   default=False, dest='max', action='store_true')
    return parser.parse_args()

def main():
    args       = parse_cl_args()
    nsols      = args.nsols
    ngens      = args.ngens
    b          = args.b
    a          = args.a
    a_step     = a / ngens
    constraints = [[-86, -80], [24, 32]]

    print("Initializing WOA...")
    start_time = time.time()
    opt_alg = WhaleOptimization(planting, constraints, nsols, b, a, a_step,
                                LonGrid, LatGrid, percover, depth_grid, args.max)

    print("Running WOA...")
    for gen in range(ngens):
        opt_alg.optimize()
        if (gen + 1) % 5 == 0:
            print(f"  Generation {gen+1}/{ngens}")

    elapsed = time.time() - start_time

    # Build all_solutions same as Trial2.py
    all_solutions = []
    for gen_positions in opt_alg.agent_paths:
        generation_data = []
        for agent_pos in gen_positions:
            x, y = agent_pos
            fitness = planting(x, y, LonGrid, LatGrid, percover, depth_grid)
            generation_data.append((fitness, (x, y)))
        all_solutions.append(generation_data)

    results = build_results(env, all_solutions, LonGrid, LatGrid, elapsed)
    print_metrics(env, results)
    plot_results(env, results, algorithm_name="WOA",
                 save_path=os.path.join(os.path.dirname(__file__), "woa_results.png"),
                 lon_grid=LonGrid, lat_grid=LatGrid)

if __name__ == '__main__':
    main()