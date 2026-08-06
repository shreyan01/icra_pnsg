"""
Visualize fitter results: for each synthetic test shape, plot the partial/
noisy input cloud alongside the fitted superquadric surface, so fit quality
can be checked by eye, not just by RMSE.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from superquadric import (
    fit_superquadric, sample_superquadric_surface,
    crop_partial_view, add_noise, local_to_world, _fexp
)
from validate_fitter import GROUND_TRUTH_SHAPES


def surface_mesh(params, n=40):
    """Dense surface grid of a superquadric for plotting (not fitting)."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, n)
    omega = np.linspace(-np.pi, np.pi, n)
    eta, omega = np.meshgrid(eta, omega)

    e1, e2 = params['eps1'], params['eps2']
    a1, a2, a3 = params['a1'], params['a2'], params['a3']

    def sc(angle, e):
        return _fexp(np.cos(angle), e)

    def ss(angle, e):
        return _fexp(np.sin(angle), e)

    x = a1 * sc(eta, e1) * sc(omega, e2)
    y = a2 * sc(eta, e1) * ss(omega, e2)
    z = a3 * ss(eta, e1)

    local = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    world = local_to_world(local, params['cx'], params['cy'], params['cz'],
                            params['roll'], params['pitch'], params['yaw'])
    X = world[:, 0].reshape(x.shape)
    Y = world[:, 1].reshape(y.shape)
    Z = world[:, 2].reshape(z.shape)
    return X, Y, Z


def plot_case(ax, name, gt_params, seed=0):
    rng = np.random.default_rng(seed)
    full_cloud = sample_superquadric_surface(gt_params, n_points=4000, rng=rng)
    partial_cloud = crop_partial_view(full_cloud, camera_dir=np.array([0.3, -0.2, 1.0]),
                                       keep_fraction=0.55)
    noisy_cloud = add_noise(partial_cloud, sigma=0.002, rng=rng)

    fitted, info = fit_superquadric(noisy_cloud, verbose=0)

    # input cloud: what the "sensor" actually saw
    ax.scatter(noisy_cloud[:, 0], noisy_cloud[:, 1], noisy_cloud[:, 2],
               s=3, c='#1F77B4', alpha=0.5, label='input cloud (partial+noisy)')

    # fitted surface: dense point scatter avoids the crossing-line artifact
    # that plot_wireframe produces on low-epsilon (boxy) shapes, where the
    # spherical-product parameterization clusters samples at corners.
    X, Y, Z = surface_mesh(fitted, n=55)
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=1.5, c='#D62728', alpha=0.35,
               label='fitted superquadric')

    ax.set_title(f'{name}\nRMSE={info["rmse"]:.4f}  eps1={fitted["eps1"]:.2f}  eps2={fitted["eps2"]:.2f}',
                 fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    names = list(GROUND_TRUTH_SHAPES.keys())
    fig = plt.figure(figsize=(14, 12))

    for i, name in enumerate(names):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        plot_case(ax, name, GROUND_TRUTH_SHAPES[name])

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4',
                   markersize=6, label='input cloud (partial + noisy)'),
        plt.Line2D([0], [0], color='#D62728', label='fitted superquadric'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10)
    fig.suptitle('Superquadric fitting: partial-view, noisy synthetic clouds vs. recovered shape',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig('/home/claude/pnsg/src/fit_visualization.png', dpi=140)
    print('saved fit_visualization.png')


if __name__ == '__main__':
    main()
