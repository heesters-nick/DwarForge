import logging
import os
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm

logger = logging.getLogger(__name__)


def plot_cutout(cutout, in_dict, figure_dir, show_plot=True, save_plot=False):
    """
    Plot cutouts in available bands.
    :param cutout: cutout data
    :param in_dict: band dictionary
    :param figure_dir: figure path
    :param show_plot: show plot
    :param save_plot: save plot
    :return: cutout plot
    """
    image_data = cutout['images']
    tile = cutout['tile']
    obj_ids = np.array([x.decode('utf-8') for x in cutout['cfis_id']])
    n_objects, n_bands = image_data.shape[0], image_data.shape[1]
    fig, axes = plt.subplots(n_objects, n_bands, figsize=(n_bands * 4, n_objects * 4))

    # Make sure axes is always a 2D array
    if n_objects == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through objects and filter bands, and plot each image
    for i in range(n_objects):  # Number of objects
        for j, band in enumerate(in_dict.keys()):  # Number of filter bands
            filter_name = in_dict[band]['band']
            ax = axes[i, j]

            # Get the image data for the current object and filter band
            image = image_data[i, j]
            # Display the image
            norm = simple_norm(image, 'sqrt', percent=98.0)
            ax.imshow(image, norm=norm, cmap='gray_r', origin='lower')  # Adjust the cmap as needed
            ax.set_title(f'{obj_ids[i]}, {filter_name}')

            # Optionally, you can turn off axis labels if they are not needed
            ax.axis('off')

    plt.tight_layout()

    if save_plot:
        plt.savefig(
            os.path.join(figure_dir, f'cutouts_tile_{tile[0]}_{tile[1]}.pdf'),
            bbox_inches='tight',
            dpi=300,
        )
    if show_plot:
        plt.show()
    else:
        plt.close()


def open_fits_files_in_ds9(file1_path, file2_path):
    try:
        # Construct the DS9 command with all the required options
        cmd = [
            'ds9',
            '-single',
            '-frame 1',
            file1_path,
            '-scale zscale',
            '-frame new',
            file2_path,
            '-scale zscale',
            '-frame lock wcs',
            '-wcs sky icrs',  # Set WCS to ICRS
            '-wcs skyformat degrees',  # Display coordinates in degrees
            '-geometry 1920x1080+0+0',
        ]

        # Run the DS9 command
        subprocess.run(cmd, check=True)
        print(
            f'Successfully opened {os.path.basename(file1_path)} and {os.path.basename(file2_path)} in DS9.'
        )
    except subprocess.CalledProcessError:
        print('Error: Failed to open the FITS files in DS9.')
    except FileNotFoundError:
        print('Error: DS9 is not installed or not in your system PATH.')


def _is_new_id(x: Any) -> bool:
    # "new" means: NaN/None/empty/"None" (string)
    if x is None:
        return True
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return True
    if isinstance(x, str) and (
        x.strip() == '' or x.strip().lower() == 'none' or x.strip().lower() == 'nan'
    ):
        return True
    return False


def plot_cutouts(
    cutouts: np.ndarray,
    ids: np.ndarray,
    coords: np.ndarray,
    mode: str = 'grid',  # "grid" or "channel"
    max_cols: int = 10,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot preprocessed image cutouts aligned with object IDs and coordinates.

    - Coordinates at bottom-right in a black semi-transparent box.
    - ID at top-left ONLY if id is not NaN (new => no ID shown).
    - Lime for new objects, orange for known.
    """
    if cutouts.ndim not in (3, 4):
        raise ValueError(f'cutouts must be (N,H,W) or (N,H,W,3); got {cutouts.shape}')
    N = cutouts.shape[0]
    if len(ids) != N:
        raise ValueError(f'len(ids)={len(ids)} does not match N={N}')
    if coords.shape != (N, 2):
        raise ValueError(f'coords must have shape (N,2); got {coords.shape}')

    C = 1 if cutouts.ndim == 3 else cutouts.shape[3]
    if C not in (1, 3):
        raise ValueError(f'Only 1 or 3 channels supported; got C={C}')

    if mode.lower() == 'grid':
        n_cols = int(min(max_cols, max(1, N)))
        n_rows = (N + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (3 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)

        for idx in range(n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            if idx < N:
                img = cutouts[idx]
                if C == 1:
                    ax.imshow(img, cmap='gray', origin='lower', aspect='equal')
                else:
                    ax.imshow(img, origin='lower', aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])

                ra, dec = coords[idx]
                obj_id = ids[idx]
                is_new = _is_new_id(obj_id)
                status_color = 'lime' if is_new else 'orange'

                # bottom-right coords in black box
                ax.text(
                    0.97,
                    0.03,
                    f'{ra:.4f}, {dec:.4f}',
                    color=status_color,
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2),
                    transform=ax.transAxes,
                    va='bottom',
                    ha='right',
                )

                # top-left ID only if NOT NaN (known object)
                if not is_new:
                    ax.text(
                        0.03,
                        0.97,
                        str(obj_id),
                        color=status_color,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, pad=2),
                        transform=ax.transAxes,
                        va='top',
                        ha='left',
                    )
            else:
                ax.axis('off')

        plt.tight_layout(pad=0.5)

    elif mode.lower() == 'channel':
        if C == 1:
            n_cols = 1
            col_titles = ['Image']
        else:
            n_cols = 4
            col_titles = ['Red', 'Green', 'Blue', 'RGB']

        if figsize is None:
            figsize = (3 * n_cols, 3 * N)

        fig, axes = plt.subplots(N, n_cols, figsize=figsize, constrained_layout=True)
        axes = np.atleast_2d(axes)

        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=12, fontweight='bold', pad=6)

        for i in range(N):
            if C == 1:
                ax = axes[i, 0]
                ax.imshow(cutouts[i], cmap='gray', origin='lower', aspect='equal')
                rgb_ax = ax
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                rgb = cutouts[i]
                # R,G,B panels
                for j in range(3):
                    axes[i, j].imshow(rgb[:, :, j], cmap='gray', origin='lower', aspect='equal')
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                # RGB composite
                axes[i, 3].imshow(rgb, origin='lower', aspect='equal')
                axes[i, 3].set_xticks([])
                axes[i, 3].set_yticks([])
                rgb_ax = axes[i, 3]

            ra, dec = coords[i]
            obj_id = ids[i]
            is_new = _is_new_id(obj_id)
            status_color = 'lime' if is_new else 'orange'

            # bottom-right coords on RGB/single panel
            rgb_ax.text(
                0.97,
                0.03,
                f'{ra:.4f}, {dec:.4f}',
                color=status_color,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=2),
                transform=rgb_ax.transAxes,
                va='bottom',
                ha='right',
            )

            # top-left ID only if known
            if not is_new:
                rgb_ax.text(
                    0.03,
                    0.97,
                    str(obj_id),
                    color=status_color,
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2),
                    transform=rgb_ax.transAxes,
                    va='top',
                    ha='left',
                )
    else:
        raise ValueError("mode must be 'grid' or 'channel'")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
