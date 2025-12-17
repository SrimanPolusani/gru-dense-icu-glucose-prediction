import os
import numpy as np
import glob
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import argparse

def compute_global_stats(data_dir, output_file):
    """
    Computes global mean and variance for dynamic features across all .npz files in a directory.

    Input:
        data_dir (str): Path to the directory containing .npz files.
        output_file (str): Path where the global statistics JSON file will be saved.

    Source:
        - Scans 'data_dir' for all files matching '*.npz'.
        - Reads the 'X_seq' key from each .npz file, which represents the dynamic feature tensor 
          with shape (batch_size, time_steps, n_features).

    Actions:
        1. Iterates through all found .npz files.
        2. For each file, extracts 'X_seq' and flattens it to (total_time_steps, n_features).
        3. Accumulates sum and sum of squares for each feature/column globally.
        4. Tracks the total count of time-steps processed across all files.
        5. Computes global mean: sum_x / total_count.
        6. Computes global variance: (sum_sq_x / total_count) - (mean^2).
        7. Handles numerical stability by setting any zero variance to 1.0.
        8. Serializes the computed statistics into a dictionary.

    Output:
        - Writes a JSON file to 'output_file' containing:
            - 'mean': List of global means for each feature.
            - 'variance': List of global variances for each feature.
            - 'n_features': Number of dynamic features processed.
            - 'count': Total number of flattened samples (time-steps) used for calculation.
    """
    files = glob.glob(os.path.join(data_dir, '*.npz'))
    if not files:
        logging.error(f"No .npz files found in {data_dir}")
        return

    logging.info(f"Found {len(files)} files. Starting statistics computation...")

    # Initialize accumulators
    try:
        first_data = np.load(files[0])
        X_seq = first_data['X_seq']

        # Standard tensorflow / keras, pytorch convention (batch_size, time_steps, features)
        n_features = X_seq.shape[2]
        logging.info(f"Detected {n_features} dynamic features.")
    except Exception as e:
        logging.error(f"Error reading first file: {e}")
        return

    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_sq_x = np.zeros(n_features, dtype=np.float64)
    total_count = 0

    for i, f in enumerate(files):
        try:
            data = np.load(f)
            X_seq = data['X_seq']
            
            # Take first patient's first time step row and then stack first patient's second timestep row veritically below it. 
            # Continue this process for first patient till the last timestep row and then vertically stack the second patient's rows below the first patient's last row and so on.
            flat_X = X_seq.reshape(-1, n_features)
            
            # axis = 0 squash all the rows into one row and sums all the values of each column. 
            sum_x += np.sum(flat_X, axis=0)
            sum_sq_x += np.sum(flat_X ** 2, axis=0)
            total_count += flat_X.shape[0]
        
            # Prints progress after processing every 100 files
            if i % 100 == 0:
                logging.info(f"Processed {i}/{len(files)} files...")
                
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")

    if total_count == 0:
        logging.error("No samples processed.")
        return

    mean = sum_x / total_count
    # Computational formula of variance E[x^2] - E[x]^2
    variance = (sum_sq_x / total_count) - (mean ** 2)
    variance[variance == 0] = 1.0
    
    logging.info("Computation complete.")
    logging.info(f"Total Time-Steps Processed: {total_count}")

    stats = {
        'mean': mean.tolist(),
        'variance': variance.tolist(),
        'n_features': n_features,
        'count': int(total_count)
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f)
        
    logging.info(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed_tensors')
    parser.add_argument('--output', type=str, default='normalization_stats.json')
    args = parser.parse_args()
    
    compute_global_stats(args.data_dir, args.output)
