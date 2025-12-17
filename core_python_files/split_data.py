import os
import shutil
import random
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(source_dir, dest_dir, split_fraction=0.1):
    """
    Moves a fraction of .npz files from source_dir to dest_dir.
    """
    if not os.path.exists(source_dir):
        logging.error(f"Source directory {source_dir} does not exist.")
        return

    if not os.path.exists(dest_dir):
        logging.info(f"Creating destination directory: {dest_dir}")
        os.makedirs(dest_dir)

    # Get all .npz files
    files = [f for f in os.listdir(source_dir) if f.endswith('.npz')]
    total_files = len(files)
    
    if total_files == 0:
        logging.warning(f"No .npz files found in {source_dir}.")
        return

    # Calculate number of files to move
    num_to_move = int(total_files * split_fraction)
    logging.info(f"Found {total_files} files. Moving {num_to_move} ({split_fraction*100}%) to {dest_dir}...")

    # Randomly select files
    files_to_move = random.sample(files, num_to_move)

    # Move files
    for f in files_to_move:
        src_path = os.path.join(source_dir, f)
        dst_path = os.path.join(dest_dir, f)
        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            logging.error(f"Error moving {f}: {e}")

    logging.info("Data split complete.")
    logging.info(f"Training files remaining: {len(os.listdir(source_dir))}")
    logging.info(f"Evaluation files: {len(os.listdir(dest_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and evaluation sets.")
    parser.add_argument('--source', type=str, default='processed_tensors', help="Source directory (Training)")
    parser.add_argument('--dest', type=str, default='eval_tensors', help="Destination directory (Evaluation)")
    parser.add_argument('--fraction', type=float, default=0.1, help="Fraction of data to move to evaluation (0.0-1.0)")
    
    args = parser.parse_args()
    
    split_data(args.source, args.dest, args.fraction)
