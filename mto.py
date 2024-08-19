import time

import mtolib.main as mto

"""Example program - using original settings"""
try:
    # Get the input image and parameters
    image, params = mto.setup()
except Exception as e:
    print(f'Error while executing mto.setup(): {e}')

try:
    # Pre-process the image
    processed_image = mto.preprocess_image(image, params, n=2)
except Exception as e:
    print(f'Error while executing mto.preprocess_image: {e}')

try:
    # Build a max tree
    mt = mto.build_max_tree(processed_image, params)
except Exception as e:
    print(f'Error while executing mto.build_max_tree: {e}')

try:
    # Filter the tree and find objects
    id_map, sig_ancs = mto.filter_tree(mt, processed_image, params)
except Exception as e:
    print(f'Error while executing mto.filter_tree: {e}')

try:
    # Relabel objects for clearer visualisation
    id_map = mto.relabel_segments(id_map, shuffle_labels=False)
except Exception as e:
    print(f'Error while executing mto.relabel_segments: {e}')

try:
    # Generate output files
    mto.generate_image(image, id_map, params)
except Exception as e:
    print(f'Error while executing mto.generate_image: {e}')

try:
    mto.generate_parameters(image, id_map, sig_ancs, params)
except Exception as e:
    print(f'Error while executing mto.generate_parameters: {e}')


# Function to measure and print execution time
def measure_time(label, func, *args):
    start_time = time.time()
    try:
        result = func(*args)
        elapsed_time = time.time() - start_time
        print(f'{label} took {elapsed_time:.4f} seconds')
        return result
    except Exception as e:
        print(f'Error while executing {label}: {e}')
        return None


# # Example program - using original settings
# image, params = measure_time('mto.setup()', mto.setup)

# processed_image = measure_time('mto.preprocess_image', mto.preprocess_image, image, params, 2)

# mt = measure_time('mto.build_max_tree', mto.build_max_tree, processed_image, params)

# id_map, sig_ancs = measure_time('mto.filter_tree', mto.filter_tree, mt, processed_image, params)

# id_map = measure_time('mto.relabel_segments', mto.relabel_segments, id_map, False)

# measure_time('mto.generate_image', mto.generate_image, image, id_map, params)

# measure_time('mto.generate_parameters', mto.generate_parameters, image, id_map, sig_ancs, params)
