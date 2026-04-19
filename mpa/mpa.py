import time
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import levy_stable
from dataclasses import dataclass, field
from typing import Optional
from environment import Environment, COVERAGE_NONE, COVERAGE_PRESENT, COVERAGE_PATCHY, COVERAGE_CONTINUOUS
from metrics import *
from plot_results import plot_results

# ─────────────────────────────────────────────
#  FISH AGENT
# ─────────────────────────────────────────────
@dataclass
class Fish:
    row: float
    col: float
    trajectory: list = field(default_factory=list)

    def pos(self) -> np.ndarray:
        return np.array([self.row, self.col])

    def record(self):
        self.trajectory.append((self.row, self.col))

    def move_to(self, new_row: float, new_col: float, env: Environment):
        self.row = float(np.clip(new_row, 0, env.rows - 1))
        self.col = float(np.clip(new_col, 0, env.cols - 1))
        self.record()

class MPASwarm:
    """
    Marine Predators Algorithm adapted for seagrass survey.

    - MPA drives global search (target selection)
    - Fish execute local movement + sensing
    - Preserves trajectories, mapping, and evaluation metrics
    """

    def __init__(
        self,
        env: Environment,
        n_agents: int = 25,
        step: float = 10.0,
        max_iter: int = 100,
        rng_seed: Optional[int] = None,
    ):
        self.env = env
        self.n_agents = n_agents
        self.step = step
        self.max_iter = max_iter
        self.rng = np.random.default_rng(rng_seed)

        # Initialize fish
        self.fish = []
        for _ in range(n_agents):
            r = self.rng.uniform(0, env.rows - 1)
            c = self.rng.uniform(0, env.cols - 1)
            f = Fish(r, c)
            f.record()
            self.fish.append(f)

        # MPA state
        self.dim = 2
        self.lb = np.array([0, 0])
        self.ub = np.array([env.rows - 1, env.cols - 1])

        self.prey = self.rng.random((n_agents, self.dim)) * (self.ub - self.lb) + self.lb
        self.fitness = np.full((n_agents, 1), np.inf)

        self.top_predator_pos = np.zeros(self.dim)
        self.top_predator_fit = np.inf

        self.fit_old = self.fitness.copy()
        self.prey_old = self.prey.copy()

        # Observation grids
        self.obs_likelihood = np.full((env.rows, env.cols), -1.0)
        self.obs_coverage   = np.full((env.rows, env.cols), -1, dtype=int)
        self.obs_depth      = np.full((env.rows, env.cols), np.nan)

        self.coverage_over_time = []

        for f in self.fish:
            self._sense_and_record(f)

    # ───────────────────────────────────────────────
    # Objective function (MPA minimizes)
    # ───────────────────────────────────────────────
    def _fobj(self, pos):
        r = int(np.clip(pos[0], 0, self.env.rows - 1))
        c = int(np.clip(pos[1], 0, self.env.cols - 1))

        likelihood = self.env.sense_likelihood(r, c)

        unexplored_bonus = 1.0 if self.obs_likelihood[r, c] < 0 else 0.0

        return -(likelihood + 0.5 * unexplored_bonus)

    # ───────────────────────────────────────────────
    # One iteration of MPA
    # ───────────────────────────────────────────────
    def _mpa_update(self, iter):
        fads = 0.2
        p = 0.5

        # Evaluate fitness
        for i in range(self.n_agents):
            self.prey[i] = np.clip(self.prey[i], self.lb, self.ub)
            self.fitness[i, 0] = self._fobj(self.prey[i])

            if self.fitness[i, 0] < self.top_predator_fit:
                self.top_predator_fit = self.fitness[i, 0]
                self.top_predator_pos = self.prey[i].copy()

        # Marine memory saving
        mask = self.fit_old < self.fitness
        self.prey[mask.flatten()] = self.prey_old[mask.flatten()]
        self.fitness[mask] = self.fit_old[mask]

        self.fit_old = self.fitness.copy()
        self.prey_old = self.prey.copy()

        elite = np.tile(self.top_predator_pos, (self.n_agents, 1))
        cf = (1 - iter / self.max_iter) ** (2 * iter / self.max_iter)

        rl = 0.05 * levy_stable.rvs(1.5, 0, size=(self.n_agents, self.dim))
        rb = self.rng.standard_normal((self.n_agents, self.dim))

        # Movement phases
        for i in range(self.n_agents):
            for j in range(self.dim):
                r = self.rng.random()

                if iter < self.max_iter / 3:
                    step = rb[i, j] * (elite[i, j] - rb[i, j] * self.prey[i, j])
                    self.prey[i, j] += p * r * step

                elif iter < 2 * self.max_iter / 3:
                    if i > self.n_agents / 2:
                        step = rb[i, j] * (rb[i, j] * elite[i, j] - self.prey[i, j])
                        self.prey[i, j] = elite[i, j] + p * cf * step
                    else:
                        step = rl[i, j] * (elite[i, j] - rl[i, j] * self.prey[i, j])
                        self.prey[i, j] += p * r * step
                else:
                    step = rl[i, j] * (rl[i, j] * elite[i, j] - self.prey[i, j])
                    self.prey[i, j] = elite[i, j] + p * cf * step

        # FADs effect
        if self.rng.random() < fads:
            u = self.rng.random((self.n_agents, self.dim)) < fads
            self.prey += cf * ((self.lb + self.rng.random((self.n_agents, self.dim)) * (self.ub - self.lb)) * u)
        else:
            r = self.rng.random()
            perm1 = self.prey[self.rng.permutation(self.n_agents)]
            perm2 = self.prey[self.rng.permutation(self.n_agents)]
            self.prey += (fads * (1 - r) + r) * (perm1 - perm2)

    # ───────────────────────────────────────────────
    # Simulation loop
    # ───────────────────────────────────────────────
    def run(self):
        n_plantable = max(int(np.sum(self.env.likelihood_grid > 0)), 1)
        start_time = time.time()

        for iter in range(self.max_iter):
            # MPA decides best global target
            self._mpa_update(iter)
            target = self.top_predator_pos

            # Fish move toward target
            for fish in self.fish:
                new_pos = self._move_toward(fish.pos(), target)
                fish.move_to(new_pos[0], new_pos[1], self.env)
                self._sense_and_record(fish)

            # Coverage tracking
            n_found = int(np.sum(
                (self.obs_likelihood > 0) & (self.env.likelihood_grid > 0)
            ))
            pct = n_found / n_plantable * 100
            self.coverage_over_time.append(pct)

        elapsed = time.time() - start_time
        return self.build_results(elapsed)

    # ───────────────────────────────────────────────
    # Utilities
    # ───────────────────────────────────────────────
    def _move_toward(self, cur, target):
        direction = target - cur
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return cur.copy()
        return cur + (direction / dist) * min(self.step, dist)

    def _sense_and_record(self, fish):
        r = int(np.clip(fish.row, 0, self.env.rows - 1))
        c = int(np.clip(fish.col, 0, self.env.cols - 1))
        self.obs_likelihood[r, c] = self.env.sense_likelihood(r, c)
        self.obs_coverage[r, c]   = self.env.sense_coverage(r, c)
        self.obs_depth[r, c]      = self.env.sense_depth(r, c)

    def build_results(self, elapsed=0.0):
        env = self.env
        n_plantable = int(np.sum(env.likelihood_grid > 0))
        n_found = int(np.sum(
            (self.obs_likelihood > 0) & (env.likelihood_grid > 0)
        ))

        coverage_pct = (n_found / n_plantable * 100) if n_plantable > 0 else 0.0
        cm, class_names = compute_confusion_matrix(env, self.obs_likelihood)

        return dict(
            observed_likelihood=self.obs_likelihood.copy(),
            observed_coverage=self.obs_coverage.copy(),
            observed_depth=self.obs_depth.copy(),

            coverage_percent=coverage_pct,
            coverage_over_time=self.coverage_over_time.copy(),

            n_plantable_found=n_found,
            n_plantable_total=n_plantable,

            rmse=compute_rmse(env, self.obs_likelihood),
            confusion_matrix=cm,
            class_names=class_names,
            computation_time=elapsed,
            trajectories=[f.trajectory.copy() for f in self.fish],
        )

if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), '..')
    LonGrid    = np.load(os.path.join(root, 'lon_grid.npy'))
    LatGrid    = np.load(os.path.join(root, 'lat_grid.npy'))
    percover   = np.load(os.path.join(root, 'percover.npy'))
    depth_grid = np.load(os.path.join(root, 'depth_grid.npy'))
    ROWS, COLS = percover.shape

    coverage_grid = np.zeros(percover.shape, dtype=int)
    coverage_grid[percover == 0.0]  = COVERAGE_CONTINUOUS
    coverage_grid[percover == 0.25] = COVERAGE_PRESENT
    coverage_grid[percover == 0.5]  = COVERAGE_PATCHY
    coverage_grid[percover == 0.55] = COVERAGE_PRESENT
    coverage_grid[percover >= 0.7]  = COVERAGE_NONE

    env = Environment(depth_grid, coverage_grid)

    # Find the viable area center to start fish near planting sites
    viable = np.where(env.likelihood_grid > 0)
    if len(viable[0]) > 0:
        start_row = float(np.mean(viable[0]))
        start_col = float(np.mean(viable[1]))
        print(f"Starting fish near viable area center: row={start_row:.0f}, col={start_col:.0f}")
    else:
        start_row = ROWS / 2
        start_col = COLS / 2

    mpa_swarm = MPASwarm(
        env          = env,
        n_agents     = 1000,
        step         = 12.0,
        max_iter     = 100,
        rng_seed     = 0,
    )

    results = mpa_swarm.run()

    print_metrics(env, results)
    plot_results(env, results, algorithm_name="MPA", save_path="mpa_results.png",
             lon_grid=LonGrid, lat_grid=LatGrid)