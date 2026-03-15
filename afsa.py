"""
afsa.py - Artificial Fish Swarm Algorithm for Seagrass Restoration
ROB 572 - McSwain, Sieh, Chen

Three core capabilities:
  1. Navigate the environment using fish behaviors (prey, swarm, follow)
  2. Map seagrass coverage by sensing and recording values at each position
  3. Classify planting likelihood per Algorithm 1 (depth + coverage -> 0/33/66/100%)

Evaluation metrics:
  - Coverage % over time (per iteration)
  - RMSE between observed and ground-truth likelihood maps
  - Confusion matrix on planting classification
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  PLANTING CLASSIFICATION  (Algorithm 1)
# ─────────────────────────────────────────────

# Coverage codes expected from your environment layer
COVERAGE_NONE       = 0   # No data / no seagrass -> treat as no coverage
COVERAGE_PRESENT    = 1   # Seagrass present
COVERAGE_PATCHY     = 2   # Seagrass patchy
COVERAGE_CONTINUOUS = 3   # Seagrass continuous

DEPTH_MIN = 1.0   # meters  (Algorithm 1 lower bound)
DEPTH_MAX = 3.0   # meters  (Algorithm 1 upper bound)


def planting_likelihood(depth: float, coverage: int) -> float:
    """
    Algorithm 1 from proposal:
      Depth must be 1-3 m, then:
        Continuous -> 0%
        Present    -> 33%
        Patchy     -> 66%
        No data    -> 100%
      Outside depth range -> 0%

    Returns a float in [0.0, 1.0].
    """
    if not (DEPTH_MIN <= depth <= DEPTH_MAX):
        return 0.0
    mapping = {
        COVERAGE_CONTINUOUS: 0.00,
        COVERAGE_PRESENT:    0.33,
        COVERAGE_PATCHY:     0.66,
        COVERAGE_NONE:       1.00,
    }
    return mapping.get(coverage, 0.0)


def likelihood_to_class(val: float) -> int:
    """
    Convert a likelihood float to a discrete class label for confusion matrix.
      0.00 -> class 0  (do not plant)
      0.33 -> class 1  (low likelihood)
      0.66 -> class 2  (medium likelihood)
      1.00 -> class 3  (high likelihood)
    """
    if val <= 0.0:
        return 0
    elif val <= 0.33:
        return 1
    elif val <= 0.66:
        return 2
    else:
        return 3


# ─────────────────────────────────────────────
#  ENVIRONMENT INTERFACE
# ─────────────────────────────────────────────

class Environment:
    """
    Thin wrapper around your existing coral.py / seagrass.py data.

    Expects two 2-D NumPy arrays of the same shape:
      depth_grid    : float  - water depth in metres at each cell
      coverage_grid : int    - one of the COVERAGE_* constants above

    Missing depth    -> filled with nearest-neighbour (Assumption 1)
    Missing coverage -> treated as COVERAGE_NONE (Assumption 2)
    """

    def __init__(self, depth_grid: np.ndarray, coverage_grid: np.ndarray):
        assert depth_grid.shape == coverage_grid.shape, \
            "depth_grid and coverage_grid must have the same shape"
        self.depth_grid    = depth_grid.astype(float)
        self.coverage_grid = coverage_grid.astype(int)
        self.rows, self.cols = depth_grid.shape

        # Pre-compute ground-truth planting likelihood for every cell
        self.likelihood_grid = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                self.likelihood_grid[r, c] = planting_likelihood(
                    self.depth_grid[r, c],
                    self.coverage_grid[r, c]
                )

    def sense_depth(self, r: int, c: int) -> float:
        r, c = self._clip(r, c)
        return self.depth_grid[r, c]

    def sense_coverage(self, r: int, c: int) -> int:
        r, c = self._clip(r, c)
        return self.coverage_grid[r, c]

    def sense_likelihood(self, r: int, c: int) -> float:
        """Food value used by AFSA - higher = better planting site."""
        r, c = self._clip(r, c)
        return self.likelihood_grid[r, c]

    def in_bounds(self, r: float, c: float) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _clip(self, r, c):
        return int(np.clip(r, 0, self.rows - 1)), int(np.clip(c, 0, self.cols - 1))

    @classmethod
    def from_your_data(cls, depth_grid, coverage_grid):
        """
        Convenience constructor - pass your arrays from coral.py / seagrass.py
        directly here.  Example:

            from coral import get_depth_grid
            from seagrass import get_coverage_grid
            env = Environment.from_your_data(get_depth_grid(), get_coverage_grid())
        """
        return cls(depth_grid, coverage_grid)


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


# ─────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────

def compute_rmse(env: Environment, obs_likelihood: np.ndarray) -> float:
    """
    RMSE between robot-observed likelihood and ground truth,
    computed only over cells the robots actually visited.
    """
    visited_mask = obs_likelihood >= 0
    if visited_mask.sum() == 0:
        return float('nan')
    diff = obs_likelihood[visited_mask] - env.likelihood_grid[visited_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_confusion_matrix(env: Environment, obs_likelihood: np.ndarray):
    """
    4x4 confusion matrix comparing observed vs ground-truth planting class
    over visited cells only.

    Classes: 0=no plant, 1=low(0.33), 2=medium(0.66), 3=high(1.0)

    Since obs_likelihood is read directly from the environment (no sensor
    noise in simulation), this confirms classification consistency and
    shows which zones were visited. When you add sensor noise later,
    this will show real misclassification rates.

    Returns:
      cm         : 4x4 int array  (rows=true, cols=predicted)
      class_names: list of label strings
    """
    visited_mask = obs_likelihood >= 0
    true_classes = np.array([
        likelihood_to_class(v) for v in env.likelihood_grid[visited_mask]
    ])
    pred_classes = np.array([
        likelihood_to_class(v) for v in obs_likelihood[visited_mask]
    ])

    n_classes = 4
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        cm[t, p] += 1

    return cm, ['No Plant (0%)', 'Low (33%)', 'Medium (66%)', 'High (100%)']


def print_metrics(env, results):
    """Print RMSE and confusion matrix to terminal."""
    rmse  = results['rmse']
    cm    = results['confusion_matrix']
    names = results['class_names']

    print(f"\n=== Evaluation Metrics ===")
    print(f"Coverage      : {results['coverage_percent']:.1f}%  "
          f"({results['n_plantable_found']} / {results['n_plantable_total']} plantable cells)")
    print(f"RMSE          : {rmse:.4f}  (0 = perfect, max = 1)")
    print(f"Compute time  : {results['computation_time']:.2f}s")
    print(f"\nConfusion Matrix (rows=true class, cols=observed class):")
    header = f"{'':>15}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row_str = f"{row_name:>15}" + "".join(f"{cm[i,j]:>14}" for j in range(4))
        print(row_str)
    print(f"\nPer-class recall (correctly identified / total true):")
    for i, name in enumerate(names):
        total   = cm[i].sum()
        correct = cm[i, i]
        pct     = (correct / total * 100) if total > 0 else 0.0
        print(f"  {name:>15}: {correct:>5} / {total:>5}  ({pct:.1f}%)")


# ─────────────────────────────────────────────
#  AFSA
# ─────────────────────────────────────────────

class AFSA:
    """
    Artificial Fish Swarm Algorithm adapted for seagrass survey.

    Capability 1 - Navigation:
        Three behaviours in priority cascade each iteration per fish:
        1. prey   - random search in visual range; if better food found, go there
        2. swarm  - move toward neighbours centre if not overcrowded (fallback)
        3. follow - move toward the best-fed neighbour (fallback)
        4. random walk - last resort if no neighbours and no better food found

    Capability 2 - Mapping:
        Each fish senses depth, coverage, and likelihood at every visited cell.
        build_results() returns the full observation grid after the run.

    Capability 3 - Planting classification:
        Uses Algorithm 1 via planting_likelihood() at every sensed cell.

    Evaluation:
        - coverage_over_time : list of coverage % after each iteration
        - rmse               : error between observed and ground-truth maps
        - confusion_matrix   : classification accuracy across 4 planting classes
        - computation_time   : wall-clock seconds for the full run
    """

    def __init__(
        self,
        env: Environment,
        n_fish: int         = 5,
        visual: float       = 20.0,
        step: float         = 10.0,
        try_number: int     = 10,
        crowd_factor: float = 0.5,
        max_iter: int       = 100,
        start_row: Optional[float] = None,
        start_col: Optional[float] = None,
        rng_seed: Optional[int]    = None,
    ):
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
            r = float(np.clip(sr + self.rng.uniform(-visual/2, visual/2),
                              0, env.rows - 1))
            c = float(np.clip(sc + self.rng.uniform(-visual/2, visual/2),
                              0, env.cols - 1))
            f = Fish(r, c)
            f.record()
            self.fish.append(f)

        # Observation grids -- -1 / NaN means unvisited
        self.obs_likelihood = np.full((env.rows, env.cols), -1.0)
        self.obs_coverage   = np.full((env.rows, env.cols), -1, dtype=int)
        self.obs_depth      = np.full((env.rows, env.cols), np.nan)

        # Per-iteration coverage % tracker
        self.coverage_over_time = []

        for f in self.fish:
            self._sense_and_record(f)

    # ── Public interface ──────────────────────────────────────────────────

    def run(self) -> dict:
        """Run AFSA for max_iter iterations. Returns full result dict."""
        n_plantable = max(int(np.sum(self.env.likelihood_grid > 0)), 1)
        start_time  = time.time()

        for iteration in range(self.max_iter):
            for fish in self.fish:
                self._step(fish)

            # Record coverage % this iteration
            n_found = int(np.sum(
                (self.obs_likelihood > 0) & (self.env.likelihood_grid > 0)
            ))
            pct = n_found / n_plantable * 100
            self.coverage_over_time.append(pct)

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter} | "
                      f"coverage: {pct:.1f}%")

        elapsed = time.time() - start_time
        return self.build_results(elapsed)

    def build_results(self, elapsed: float = 0.0) -> dict:
        """Returns mapping outputs, coverage curve, RMSE, and confusion matrix."""
        env         = self.env
        n_plantable = int(np.sum(env.likelihood_grid > 0))
        n_found     = int(np.sum(
            (self.obs_likelihood > 0) & (env.likelihood_grid > 0)
        ))
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

    # ── Internal: AFSA behaviours ─────────────────────────────────────────

    def _step(self, fish: Fish):
        env     = self.env
        cur_pos = fish.pos()
        cur_val = env.sense_likelihood(int(cur_pos[0]), int(cur_pos[1]))
        neighbours = self._get_neighbours(fish)

        # ── Behaviour 1: Prey (FIRST PRIORITY) ───────────────────────────
        # Scan random positions in visual range; move to best food found.
        best_prey_val = cur_val
        best_prey_pos = None
        for _ in range(self.try_number):
            angle  = self.rng.uniform(0, 2 * np.pi)
            radius = self.rng.uniform(0, self.visual)
            rp = np.array([
                cur_pos[0] + radius * np.sin(angle),
                cur_pos[1] + radius * np.cos(angle),
            ])
            rp[0] = np.clip(rp[0], 0, env.rows - 1)
            rp[1] = np.clip(rp[1], 0, env.cols - 1)
            val = env.sense_likelihood(int(rp[0]), int(rp[1]))
            if val > best_prey_val:
                best_prey_val = val
                best_prey_pos = rp

        if best_prey_pos is not None:
            target_pos = best_prey_pos

        # ── Behaviour 2: Swarm (fallback if prey found nothing) ───────────
        elif neighbours:
            centre    = np.mean([n.pos() for n in neighbours], axis=0)
            n_count   = len(neighbours)
            swarm_val = env.sense_likelihood(int(centre[0]), int(centre[1]))

            if n_count < self.n_fish * self.crowd_factor and swarm_val > cur_val:
                target_pos = centre

            # ── Behaviour 3: Follow (fallback if swarm also unhelpful) ────
            else:
                best_n = max(
                    neighbours,
                    key=lambda n: env.sense_likelihood(int(n.row), int(n.col))
                )
                best_n_val   = env.sense_likelihood(int(best_n.row), int(best_n.col))
                best_n_count = sum(
                    1 for n in neighbours
                    if env.sense_likelihood(int(n.row), int(n.col)) >= best_n_val
                )
                if (best_n_val > cur_val and
                        best_n_count < self.n_fish * self.crowd_factor):
                    target_pos = best_n.pos()
                else:
                    angle      = self.rng.uniform(0, 2 * np.pi)
                    target_pos = cur_pos + self.step * np.array(
                        [np.sin(angle), np.cos(angle)]
                    )

        # ── Random walk (last resort) ─────────────────────────────────────
        else:
            angle      = self.rng.uniform(0, 2 * np.pi)
            target_pos = cur_pos + self.step * np.array(
                [np.sin(angle), np.cos(angle)]
            )

        new_pos = self._move_toward(cur_pos, target_pos)
        fish.move_to(new_pos[0], new_pos[1], env)
        self._sense_and_record(fish)

    def _move_toward(self, cur: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - cur
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return cur.copy()
        return cur + (direction / dist) * min(self.step, dist)

    def _get_neighbours(self, fish: Fish) -> list:
        return [
            other for other in self.fish
            if other is not fish and
               np.linalg.norm(other.pos() - fish.pos()) <= self.visual
        ]

    def _sense_and_record(self, fish: Fish):
        r = int(np.clip(fish.row, 0, self.env.rows - 1))
        c = int(np.clip(fish.col, 0, self.env.cols - 1))
        self.obs_likelihood[r, c] = self.env.sense_likelihood(r, c)
        self.obs_coverage[r, c]   = self.env.sense_coverage(r, c)
        self.obs_depth[r, c]      = self.env.sense_depth(r, c)


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def plot_results(env: Environment, results: dict, save_path: Optional[str] = None,
                 lon_grid: Optional[np.ndarray] = None,
                 lat_grid: Optional[np.ndarray] = None):
    """Six-panel summary figure including coverage curve and confusion matrix."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    use_geo = lon_grid is not None and lat_grid is not None

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("AFSA - Seagrass Restoration Survey", fontsize=14, fontweight='bold')

    proj = ccrs.PlateCarree() if use_geo else None

    def make_ax(pos):
        if use_geo:
            return fig.add_subplot(2, 3, pos, projection=proj)
        else:
            return fig.add_subplot(2, 3, pos)

    def add_coast(ax):
        if use_geo:
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                           lat_grid.min(), lat_grid.max()],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
            ax.add_feature(cfeature.STATES, linewidth=0.4)

    # ── Panel 1: Ground-truth planting likelihood ─────────────────────────
    ax1 = make_ax(1)
    if use_geo:
        im = ax1.pcolormesh(lon_grid, lat_grid, env.likelihood_grid,
                            transform=ccrs.PlateCarree(), cmap='YlGn',
                            vmin=0, vmax=1, zorder=1)
        add_coast(ax1)
        plt.colorbar(im, ax=ax1, label='Planting Likelihood')
        ax1.set_title('Ground Truth - Planting Likelihood')
    else:
        im = ax1.imshow(env.likelihood_grid, origin='upper', cmap='YlGn', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1, label='Planting Likelihood')
        ax1.set_title('Ground Truth - Planting Likelihood')
        ax1.set_xlabel('Column'); ax1.set_ylabel('Row')

    # ── Panel 2: Robot-observed map ───────────────────────────────────────
    ax2 = make_ax(2)
    display = np.where(results['observed_likelihood'] >= 0,
                       results['observed_likelihood'], np.nan)
    if use_geo:
        im2 = ax2.pcolormesh(lon_grid, lat_grid, display,
                             transform=ccrs.PlateCarree(), cmap='YlGn',
                             vmin=0, vmax=1, zorder=1)
        add_coast(ax2)
        plt.colorbar(im2, ax=ax2, label='Observed Likelihood')
    else:
        im2 = ax2.imshow(display, origin='upper', cmap='YlGn', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label='Observed Likelihood')
        ax2.set_xlabel('Column'); ax2.set_ylabel('Row')
    ax2.set_title(
        f"Robot-Built Map  ({results['coverage_percent']:.1f}% plantable found)\n"
        f"{results['n_plantable_found']} / {results['n_plantable_total']} cells  |  "
        f"RMSE: {results['rmse']:.4f}"
    )

    # ── Panel 3: Fish trajectories ────────────────────────────────────────
    ax3 = make_ax(3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['trajectories'])))
    if use_geo:
        # Plot likelihood as background
        ax3.pcolormesh(lon_grid, lat_grid, env.likelihood_grid,
                       transform=ccrs.PlateCarree(), cmap='Greys',
                       alpha=0.4, zorder=1)
        add_coast(ax3)
        for idx, traj in enumerate(results['trajectories']):
            if len(traj) > 1:
                traj_lons = [lon_grid[int(p[0]), int(p[1])] for p in traj]
                traj_lats = [lat_grid[int(p[0]), int(p[1])] for p in traj]
                ax3.plot(traj_lons, traj_lats, color=colors[idx], lw=1.2,
                         transform=ccrs.PlateCarree(), label=f'Fish {idx+1}', zorder=2)
                ax3.scatter(traj_lons[0], traj_lats[0], color=colors[idx],
                            marker='o', s=40, zorder=5, transform=ccrs.PlateCarree())
                ax3.scatter(traj_lons[-1], traj_lats[-1], color=colors[idx],
                            marker='*', s=80, zorder=5, transform=ccrs.PlateCarree())
    else:
        ax3.imshow(env.likelihood_grid, origin='upper', cmap='Greys', alpha=0.4)
        for idx, traj in enumerate(results['trajectories']):
            if len(traj) > 1:
                rows = [p[0] for p in traj]
                cols = [p[1] for p in traj]
                ax3.plot(cols, rows, color=colors[idx], lw=1.2, label=f'Fish {idx+1}')
                ax3.scatter(cols[0],  rows[0],  color=colors[idx], marker='o', s=40, zorder=5)
                ax3.scatter(cols[-1], rows[-1], color=colors[idx], marker='*', s=80, zorder=5)
        ax3.set_xlabel('Column'); ax3.set_ylabel('Row')
    ax3.set_title('Fish Trajectories  (o start  * end)')
    ax3.legend(fontsize=7, loc='upper right')

    # ── Panel 4: Coverage % over time ────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)  # always regular axis
    iters = range(1, len(results['coverage_over_time']) + 1)
    ax4.plot(iters, results['coverage_over_time'], color='steelblue', lw=2)
    ax4.set_title('Coverage Efficiency Over Time')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Plantable Cells Found (%)')
    ax4.set_ylim(0, max(results['coverage_over_time']) * 1.15 + 0.5)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(results['coverage_percent'], color='red', linestyle='--',
                alpha=0.6, label=f"Final: {results['coverage_percent']:.1f}%")
    ax4.legend(fontsize=8)

    # ── Panel 5: Confusion matrix ─────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    cm  = results['confusion_matrix']
    im3 = ax5.imshow(cm, cmap='Blues')
    plt.colorbar(im3, ax=ax5)
    names_short = ['No Plant', 'Low', 'Medium', 'High']
    ax5.set_xticks(range(4)); ax5.set_xticklabels(names_short, fontsize=8)
    ax5.set_yticks(range(4)); ax5.set_yticklabels(names_short, fontsize=8)
    ax5.set_xlabel('Observed Class')
    ax5.set_ylabel('True Class')
    ax5.set_title(f'Confusion Matrix\n(RMSE = {results["rmse"]:.4f})')
    for i in range(4):
        for j in range(4):
            ax5.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=9)

    # ── Panel 6: Seagrass coverage ground truth ───────────────────────────
    ax6 = make_ax(6)
    cmap4  = plt.cm.colors.ListedColormap(['#cccccc', '#ffffb2', '#74c476', '#006d2c'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = plt.cm.colors.BoundaryNorm(bounds, cmap4.N)
    if use_geo:
        ax6.pcolormesh(lon_grid, lat_grid, env.coverage_grid,
                       transform=ccrs.PlateCarree(), cmap=cmap4,
                       norm=norm, zorder=1)
        add_coast(ax6)
    else:
        ax6.imshow(env.coverage_grid, origin='upper', cmap=cmap4, norm=norm)
        ax6.set_xlabel('Column'); ax6.set_ylabel('Row')
    patches = [
        mpatches.Patch(color='#cccccc', label='None / No data'),
        mpatches.Patch(color='#ffffb2', label='Present'),
        mpatches.Patch(color='#74c476', label='Patchy'),
        mpatches.Patch(color='#006d2c', label='Continuous'),
    ]
    ax6.legend(handles=patches, fontsize=7, loc='upper right')
    ax6.set_title('Seagrass Coverage (ground truth)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  QUICK DEMO  (replace arrays with your data)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running AFSA on real Florida coast environment ...")
    LonGrid = np.load('lon_grid.npy')
    LatGrid = np.load('lat_grid.npy')

    # ── Load real environment from Mapping_of_Viable_Sites.py output ────
    percover   = np.load('percover.npy')
    depth_grid = np.load('depth_grid.npy')
    ROWS, COLS = percover.shape

    # Convert percover scores to coverage codes for Algorithm 1
    coverage_grid = np.zeros(percover.shape, dtype=int)
    coverage_grid[percover == 0.0]  = COVERAGE_CONTINUOUS  # already has seagrass
    coverage_grid[percover == 0.25] = COVERAGE_PRESENT      # 51-100% coverage
    coverage_grid[percover == 0.5]  = COVERAGE_PATCHY       # patchy/discontinuous
    coverage_grid[percover == 0.55] = COVERAGE_PRESENT      # 1-89% coverage
    coverage_grid[percover >= 0.7]  = COVERAGE_NONE         # bare/unknown = plant here

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

    afsa = AFSA(
        env          = env,
        n_fish       = 5,
        visual       = 25.0,
        step         = 12.0,
        try_number   = 10,
        crowd_factor = 0.5,
        max_iter     = 100,
        start_row    = start_row,
        start_col    = start_col,
        rng_seed     = 0,
    )

    print("Fish initialised. Running 100 iterations ...")
    results = afsa.run()

    print_metrics(env, results)
    plot_results(env, results, save_path="afsa_results.png",
             lon_grid=LonGrid, lat_grid=LatGrid)