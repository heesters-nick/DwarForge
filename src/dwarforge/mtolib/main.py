"""High level processes for MTObjects."""
# TODO rename?

from ctypes import c_double, c_float

import numpy as np

from dwarforge.mtolib import _ctype_classes as ct
from dwarforge.mtolib import maxtree
from dwarforge.mtolib.io_mto import (  # noqa: F401
    generate_image,
    generate_parameters,
    make_parser,
    read_fits_file,
)
from dwarforge.mtolib.postprocessing import relabel_segments  # noqa: F401
from dwarforge.mtolib.preprocessing import preprocess_image  # noqa: F401
from dwarforge.mtolib.tree_filtering import (  # noqa: F401
    filter_tree,
    get_c_significant_nodes,
    init_double_filtering,
)
from dwarforge.mtolib.utils import time_function


def setup():
    """Read in a file and parameters; run initialisation functions."""

    # Parse command line arguments
    p = make_parser().parse_args()

    # Warn if using default soft bias
    if p.soft_bias is None:
        p.soft_bias = 0.0
    img = read_fits_file(p.filename)

    if p.verbosity:
        print('\n---Image dimensions---')
        print('Height = ', img.shape[0])
        print('Width = ', img.shape[1])
        print('Size = ', img.size)

    # Set the pixel type based on the type in the image
    p.d_type = c_float
    if np.issubdtype(img.dtype, np.float64):
        p.d_type = c_double
        init_double_filtering(p)

    # Initialise CTypes classes
    ct.init_classes(p.d_type)

    return img, p


def max_tree_timed(img, params, maxtree_class):
    """Build and return a maxtree of a given class"""
    if params.verbosity:
        print('\n---Building Maxtree---')
    mt = maxtree_class(img, params.verbosity, params)
    mt.flood()
    return mt


def build_max_tree(img, params, maxtree_class=maxtree.OriginalMaxTree):
    return time_function(
        max_tree_timed, (img, params, maxtree_class), params.verbosity, 'create max tree'
    )
