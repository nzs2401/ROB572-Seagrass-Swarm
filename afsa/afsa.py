import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment import Environment, COVERAGE_NONE, COVERAGE_PRESENT, COVERAGE_PATCHY, COVERAGE_CONTINUOUS
from metrics import compute_rmse, compute_confusion_matrix, print_metrics
from plot_results import plot_results


@dataclass
class Fish:
    row: float
    col: float
    trajectory: list = field(default_factory=list)

    def pos(self):
        return np.array([self.row, self.col])

    def record(self):
        self.trajectory.append((self.row, self.col))

    def move_to(self, new_row, new_col, env):
        self.row = float(np.clip(new_row, 0, env.rows - 1))
        self.col = float(np.clip(new_col, 0, env.cols - 1))
        self.record()


class AFSA:
    def __init__(self, env, n_fish=10, visual=20.0, step=10.0,
                 try_number=10, crowd_factor=0.5, max_iter=100,
                 start_row=None, start_col=None, rng_seed=None):
        self.env          = env
        self.n_fish       = n_fish
        self.visual       = visual
        self.step         = step
        self.try_number   = try_number
        self.crowd_factor = crowd_factor
        self.max_iter     = max_iter
        self.rng          = np.random.default_rng(rng_seed)

        sr = start_row if start_row is not None else env.rows / 2
        sc = start_col if start_col is not None else env.cols / 2

        self.fish = []
        for _ in range(n_fish):
            r = float(np.clip(sr + self.rng.uniform(-visual/2, visual/2), 0, env.rows-1))
            c = float(np.clip(sc + self.rng.uniform(-visual/2, visual/2), 0, env.cols-1))
            f = Fish(r, c)
            f.record()
            self.fish.append(f)

        self.obs_likelihood = np.full((env.rows, env.cols), -1.0)
        self.obs_coverage   = np.full((env.rows, env.cols), -1, dtype=int)
        self.obs_depth      = np.full((env.rows, env.cols), np.nan)
        self.coverage_over_time = []

        for f in self.fish:
            self._sense_and_record(f)

    def run(self):
        n_plantable = max(int(np.sum(self.env.likelihood_grid > 0)), 1)
        start_time  = time.time()

        for iteration in range(self.max_iter):
            for fish in self.fish:
                self._step(fish)
            n_found = int(np.sum((self.obs_likelihood > 0) & (self.env.likelihood_grid > 0)))
            pct = n_found / n_plantable * 100
            self.coverage_over_time.append(pct)
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter} | coverage: {pct:.1f}%")

        return self.build_results(time.time() - start_time)

    def build_results(self, elapsed=0.0):
        env         = self.env
        n_plantable = int(np.sum(env.likelihood_grid > 0))
        n_found     = int(np.sum((self.obs_likelihood > 0) & (env.likelihood_grid > 0)))
        coverage_pct = (n_found / n_plantable * 100) if n_plantable > 0 else 0.0
        cm, class_names = compute_confusion_matrix(env, self.obs_likelihood)
        return dict(
            observed_likelihood = self.obs_likelihood.copy(),
            observed_coverage   = self.obs_coverage.copy(),
            observed_depth      = self.obs_depth.copy(),
            coverage_percent    = coverage_pct,
            coverage_over_time  = self.coverage_over_time.copy(),
            n_plantable_found   = n_found,
            n_plantable_total   = n_plantable,
            rmse                = compute_rmse(env, self.obs_likelihood),
            confusion_matrix    = cm,
            class_names         = class_names,
            computation_time    = elapsed,
            trajectories        = [f.trajectory.copy() for f in self.fish],
            fish_final_pos      = [(f.row, f.col) for f in self.fish],
        )

    def _step(self, fish):
        env     = self.env
        cur_pos = fish.pos()
        cur_val = env.sense_likelihood(int(cur_pos[0]), int(cur_pos[1]))
        neighbours = self._get_neighbours(fish)

        best_prey_val = cur_val
        best_prey_pos = None
        for _ in range(self.try_number):
            angle  = self.rng.uniform(0, 2 * np.pi)
            radius = self.rng.uniform(0, self.visual)
            rp = np.clip([cur_pos[0] + radius * np.sin(angle),
                          cur_pos[1] + radius * np.cos(angle)],
                         [0, 0], [env.rows-1, env.cols-1])
            val = env.sense_likelihood(int(rp[0]), int(rp[1]))
            if val > best_prey_val:
                best_prey_val = val
                best_prey_pos = rp

        if best_prey_pos is not None:
            target_pos = best_prey_pos
        elif neighbours:
            centre    = np.mean([n.pos() for n in neighbours], axis=0)
            n_count   = len(neighbours)
            swarm_val = env.sense_likelihood(int(centre[0]), int(centre[1]))
            if n_count < self.n_fish * self.crowd_factor and swarm_val > cur_val:
                target_pos = centre
            else:
                best_n = max(neighbours,
                             key=lambda n: env.sense_likelihood(int(n.row), int(n.col)))
                best_n_val = env.sense_likelihood(int(best_n.row), int(best_n.col))
                best_n_count = sum(1 for n in neighbours
                                   if env.sense_likelihood(int(n.row), int(n.col)) >= best_n_val)
                if best_n_val > cur_val and best_n_count < self.n_fish * self.crowd_factor:
                    target_pos = best_n.pos()
                else:
                    angle      = self.rng.uniform(0, 2 * np.pi)
                    target_pos = cur_pos + self.step * np.array([np.sin(angle), np.cos(angle)])
        else:
            angle      = self.rng.uniform(0, 2 * np.pi)
            target_pos = cur_pos + self.step * np.array([np.sin(angle), np.cos(angle)])

        new_pos = self._move_toward(cur_pos, target_pos)
        fish.move_to(new_pos[0], new_pos[1], env)
        self._sense_and_record(fish)

    def _move_toward(self, cur, target):
        direction = target - cur
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return cur.copy()
        return cur + (direction / dist) * min(self.step, dist)

    def _get_neighbours(self, fish):
        return [o for o in self.fish
                if o is not fish and np.linalg.norm(o.pos() - fish.pos()) <= self.visual]

    def _sense_and_record(self, fish):
        r = int(np.clip(fish.row, 0, self.env.rows-1))
        c = int(np.clip(fish.col, 0, self.env.cols-1))
        self.obs_likelihood[r, c] = self.env.sense_likelihood(r, c)
        self.obs_coverage[r, c]   = self.env.sense_coverage(r, c)
        self.obs_depth[r, c]      = self.env.sense_depth(r, c)


if __name__ == "__main__":
    print("Loading environment...")
    root = os.path.join(os.path.dirname(__file__), '..')
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

    viable = np.where(env.likelihood_grid > 0)
    start_row = float(np.mean(viable[0])) if len(viable[0]) > 0 else env.rows / 2
    start_col = float(np.mean(viable[1])) if len(viable[0]) > 0 else env.cols / 2

    # change parameters here to see convergence...
    afsa = AFSA(env, n_fish=20, visual=75.0, step=40.0,
            try_number=4, crowd_factor=0.3, max_iter=200,
            start_row=start_row, start_col=start_col, rng_seed=0)

    print("Running AFSA...")
    results = afsa.run()
    print_metrics(env, results)
    plot_results(env, results, algorithm_name="AFSA",
                 save_path=os.path.join(os.path.dirname(__file__), "afsa_results.png"),
                 lon_grid=LonGrid, lat_grid=LatGrid)