import pandas as pd
import sqlite3
import os
import glob
from multiprocessing import Pool

SOURCE_FOLDER = ""
OUTPUT_FOLDER = ""

# Create output folder if not existed
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def convert_single_file(file_path):
    # Setup filenames
    file_name = os.path.basename(file_path)
    table_name = file_name.replace('.csv', '')

    # Define .db names for each file that we are converting
    db_name = f"{table_name}.db"
    db_path = os.path.join(SOURCE_FOLDER, db_name)

    # Connect to the sqlite3 server
    conn = sqlite3.connect(db_path)

    # Process in chunks (Low memory use)
    # chucksize=100000 keeps RAM usage 500 MB per core (2-0.5 = leaves 1.5GB space)
    chunk_size = 100000

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunk.to_sql(table_name, con=conn, if_exits='append', index=False)

    # Close the sqlite3 connection after using
    conn.close()
    return f"The conversion is successful {file_name} --> {db_name}"

if __name__ == '__main__':
    print("Scanning source folder for files...")

    # 1. Find all CSVs
    csv_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.csv"))
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files")

    available_cores = os.cpu_count()
    cores_in_use = min(available_cores, total_files)

    print(f"Starting Parallel Processing with {cores_in_use} Cores...")
    print("-"*40)

    # Create a Pool and run
    with Pool(processes=cores_in_use) as pool:
        results = pool.map(convert_single_file, csv_files)

    # Report Results
    print("-"*40)
    print("JOB COMPLETE. Summary:")
    for line in results:
        print(line)





