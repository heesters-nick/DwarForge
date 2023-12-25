import mtolib.main as mto
import logging
import time

# Configure logging
logging.basicConfig(filename='time_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_elapsed_time(start_time, function_name):
    elapsed_time = time.time() - start_time
    logging.info(f"{function_name} - Elapsed time: {elapsed_time:.4f} seconds")

"""Example program - using original settings"""

try:
	# Get the input image and parameters
	start_time = time.time()
	image, params = mto.setup()
	log_elapsed_time(start_time, "setup")

	# Pre-process the image
	start_time = time.time()
	processed_image = mto.preprocess_image(image, params, n=2)
	log_elapsed_time(start_time, "preprocess")

	# Build a max tree
	start_time = time.time()
	mt = mto.build_max_tree(processed_image, params)
	log_elapsed_time(start_time, "build_max_tree")

	# Filter the tree and find objects
	start_time = time.time()
	id_map, sig_ancs = mto.filter_tree(mt, processed_image, params)
	log_elapsed_time(start_time, "filter_tree")

	# Relabel objects for clearer visualisation
	start_time = time.time()
	id_map = mto.relabel_segments(id_map, shuffle_labels=False)
	log_elapsed_time(start_time, "relabel_segments")

	# Generate output files
	start_time = time.time()
	#mto.generate_image(image, id_map, params)
	log_elapsed_time(start_time, "generate_image")

	start_time = time.time()
	mto.generate_parameters(image, id_map, sig_ancs, params)
	log_elapsed_time(start_time, "generate_parameters")
except Exception as e:
	print(e)
