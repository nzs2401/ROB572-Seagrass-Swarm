import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_results(env, results, algorithm_name="Algorithm", save_path=None,
                 lon_grid=None, lat_grid=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    use_geo = lon_grid is not None and lat_grid is not None
    proj    = ccrs.PlateCarree() if use_geo else None

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{algorithm_name} - Seagrass Restoration Survey  |  "
             f"Compute Time: {results['computation_time']:.2f}s",
             fontsize=14, fontweight='bold')

    def make_ax(pos):
        return fig.add_subplot(2, 2, pos, projection=proj) if use_geo \
            else fig.add_subplot(2, 2, pos)

    def add_coast(ax):
        if use_geo:
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                           lat_grid.min(), lat_grid.max()],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
            ax.add_feature(cfeature.STATES, linewidth=0.4)

    # Panel 1: Ground truth
    ax1 = make_ax(1)
    if use_geo:
        im = ax1.pcolormesh(lon_grid, lat_grid, env.likelihood_grid,
                            transform=ccrs.PlateCarree(), cmap='YlGn', vmin=0, vmax=1, zorder=1)
        add_coast(ax1)
    else:
        im = ax1.imshow(env.likelihood_grid, origin='upper', cmap='YlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, label='Planting Likelihood')
    ax1.set_title('Ground Truth - Planting Likelihood')

    # Panel 2: Robot-observed map
    ax2 = make_ax(2)
    display = np.where(results['observed_likelihood'] >= 0,
                       results['observed_likelihood'], np.nan)
    if use_geo:
        im2 = ax2.pcolormesh(lon_grid, lat_grid, display,
                             transform=ccrs.PlateCarree(), cmap='YlGn', vmin=0, vmax=1, zorder=1)
        add_coast(ax2)
    else:
        im2 = ax2.imshow(display, origin='upper', cmap='YlGn', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Observed Likelihood')
    ax2.set_title(f"Robot-Built Map  ({results['coverage_percent']:.1f}% plantable found)\n"
                  f"{results['n_plantable_found']} / {results['n_plantable_total']} cells  |  "
                  f"RMSE: {results['rmse']:.4f}")

    # # Panel 3: Trajectories (fancy)
    # ax3 = make_ax(3)
    # colors = plt.cm.tab10(np.linspace(0, 1, len(results['trajectories'])))
    # if use_geo:
    #     ax3.pcolormesh(lon_grid, lat_grid, env.likelihood_grid,
    #                    transform=ccrs.PlateCarree(), cmap='Greys', alpha=0.4, zorder=1)
    #     add_coast(ax3)
    #     for idx, traj in enumerate(results['trajectories']):
    #         if len(traj) > 1:
    #             traj_lons = [lon_grid[int(p[0]), int(p[1])] for p in traj]
    #             traj_lats = [lat_grid[int(p[0]), int(p[1])] for p in traj]
    #             ax3.plot(traj_lons, traj_lats, color=colors[idx], lw=1.2,
    #                      transform=ccrs.PlateCarree(), zorder=2)
    #             ax3.scatter(traj_lons[0], traj_lats[0], color=colors[idx],
    #                         marker='o', s=40, zorder=5, transform=ccrs.PlateCarree())
    #             ax3.scatter(traj_lons[-1], traj_lats[-1], color=colors[idx],
    #                         marker='*', s=80, zorder=5, transform=ccrs.PlateCarree())
    # else:
    #     ax3.imshow(env.likelihood_grid, origin='upper', cmap='Greys', alpha=0.4)
    #     for idx, traj in enumerate(results['trajectories']):
    #         if len(traj) > 1:
    #             rows = [p[0] for p in traj]
    #             cols = [p[1] for p in traj]
    #             ax3.plot(cols, rows, color=colors[idx], lw=1.2, label=f'Agent {idx+1}')
    #             ax3.scatter(cols[0],  rows[0],  color=colors[idx], marker='o', s=40, zorder=5)
    #             ax3.scatter(cols[-1], rows[-1], color=colors[idx], marker='*', s=80, zorder=5)
    # ax3.set_title('Agent Trajectories  (o start  * end)')
    # ax3.legend(fontsize=7, loc='upper right')

    # Panel 4: Coverage Efficiency
    ax4 = fig.add_subplot(2, 2, 3)
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

    # Panel 5: Normalized confusion matrix
    ax5 = fig.add_subplot(2, 2, 4)
    cm  = results['confusion_matrix']

    # Normalize by row totals
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide by zero
    cm_norm = cm_norm / row_sums

    im3 = ax5.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax5, label='Recall')
    names_short = ['No Plant', 'Low', 'Medium', 'High']
    ax5.set_xticks(range(4)); ax5.set_xticklabels(names_short, fontsize=8)
    ax5.set_yticks(range(4)); ax5.set_yticklabels(names_short, fontsize=8)
    ax5.set_xlabel('Observed Class')
    ax5.set_ylabel('True Class')
    ax5.set_title(f'Normalized Confusion Matrix\n(RMSE = {results["rmse"]:.4f})')
    for i in range(4):
        for j in range(4):
            ax5.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=9)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()