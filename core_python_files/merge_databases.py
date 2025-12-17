import sqlite3
import os
import glob
import time
import shutil
from concurrent.futures import ProcessPoolExecutor

# HPC RAM Disk Location (Linux)
# If on Windows/Local, falls back to standard temp dir, which might not be RAM.
RAM_DISK_DIR = "/dev/shm" if os.path.exists("/dev/shm") else os.environ.get('TEMP', '/tmp')

def copy_file_to_ram(args):
    """Worker function to copy a single file to RAM."""
    src_path, dest_dir = args
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    try:
        shutil.copy2(src_path, dest_path)
        return f"Copied {filename}"
    except Exception as e:
        return f"Error copying {filename}: {e}"

def merge_databases(source_folder, output_file, use_ram_disk=True, max_workers=8):
    """
    Optimized merge using RAM Disk and Parallel Staging.
    """
    total_start = time.time()
    
    # 1. Setup RAM Workspace
    if use_ram_disk:
        work_dir = os.path.join(RAM_DISK_DIR, "eicu_merge_work_" + str(int(time.time())))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        print(f"Using RAM Disk workspace: {work_dir}")
    else:
        work_dir = source_folder
        print("RAM Disk disabled. Using source folder directly (Slower).")

    # 2. Parallel Staging (Copy Source -> RAM)
    db_files = glob.glob(os.path.join(source_folder, "*.db"))
    ram_db_files = []
    
    if use_ram_disk:
        print(f"Staging {len(db_files)} databases to RAM using {max_workers} cores...")
        copy_args = [(f, work_dir) for f in db_files]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(copy_file_to_ram, copy_args))
            
        # Update db_files list to point to RAM paths
        ram_db_files = glob.glob(os.path.join(work_dir, "*.db"))
    else:
        ram_db_files = db_files

    # 3. Perform Merge (in RAM)
    # We write the output to RAM first for max speed
    temp_output_file = os.path.join(work_dir, "eicu_merged_temp.sqlite3")
    
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    print(f"Merging into temporary RAM file: {temp_output_file}")
    main_conn = sqlite3.connect(temp_output_file)
    
    # --- AGGRESSIVE OPTIMIZATIONS ---
    main_conn.execute("PRAGMA synchronous = OFF")
    main_conn.execute("PRAGMA journal_mode = OFF")
    main_conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    main_conn.execute("PRAGMA temp_store = MEMORY")
    main_conn.execute("PRAGMA cache_size = -6400000") # 6GB Cache
    main_conn.execute("PRAGMA mmap_size = 60000000000") # Memory map 60GB
    
    main_cursor = main_conn.cursor()

    for db_path in ram_db_files:
        filename = os.path.basename(db_path)
        table_name = os.path.splitext(filename)[0]
        
        try:
            main_cursor.execute(f"ATTACH DATABASE '{db_path}' AS src_db")
            
            # Verify table name
            main_cursor.execute(f"SELECT name FROM src_db.sqlite_master WHERE type='table' AND name='{table_name}'")
            if not main_cursor.fetchone():
                main_cursor.execute("SELECT name FROM src_db.sqlite_master WHERE type='table'")
                res = main_cursor.fetchone()
                if res:
                    table_name = res[0]
                else:
                    print(f"Skipping empty DB: {filename}")
                    main_cursor.execute("DETACH DATABASE src_db")
                    # Cleanup even if empty
                    if use_ram_disk:
                        os.remove(db_path)
                    continue

            dest_table_name = table_name.replace('.db', '').replace('.', '_')
            
            print(f"  Merging table: {dest_table_name}...")
            main_cursor.execute(f'CREATE TABLE main."{dest_table_name}" AS SELECT * FROM src_db."{table_name}"')
            main_cursor.execute("DETACH DATABASE src_db")
            
            # OPTIMIZATION: Delete source DB from RAM immediately to free space for Output DB
            if use_ram_disk:
                os.remove(db_path)
            
        except Exception as e:
            print(f"  Error merging {filename}: {e}")

    # 4. Create Indices (Sequential but fast in RAM)
    print("Creating indices (in RAM)...")
    main_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = main_cursor.fetchall()
    for (tbl,) in tables:
        main_cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_pid ON {tbl} (patientunitstayid)")
    
    main_conn.commit()
    main_conn.close()

    # 5. Final Write (RAM -> Disk)
    print(f"Saving final merged file to: {output_file}")
    shutil.move(temp_output_file, output_file)
    
    # Cleanup RAM
    if use_ram_disk:
        print("Cleaning up RAM disk...")
        shutil.rmtree(work_dir)

    print(f"TOTAL TIME: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    # Configuration
    SOURCE_FOLDER = "data" 
    OUTPUT_FILE = "eicu_merged.sqlite3"
    
    # HPC Configuration
    USE_RAM_DISK = True 
    
    # Robust Core Detection for Slurm
    try:
        # Try to get Slurm allocation
        CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    except (ValueError, TypeError):
        CORES = os.cpu_count()
        
    # Safety Check: Ensure Output is NOT in RAM Disk (otherwise it vanishes)
    if USE_RAM_DISK and OUTPUT_FILE.startswith(RAM_DISK_DIR):
        print(f"WARNING: Output file {OUTPUT_FILE} is in RAM Disk. It will be lost after job ends!")
        print("Please specify a persistent storage path for OUTPUT_FILE.")
        # We don't exit, just warn, in case the user knows what they are doing (e.g. staging elsewhere)

    print(f"--- Job Configuration ---")
    print(f"Source: {SOURCE_FOLDER}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"RAM Disk: {'Enabled (' + RAM_DISK_DIR + ')' if USE_RAM_DISK else 'Disabled'}")
    print(f"Cores: {CORES}")
    print(f"-------------------------")

    # Create a dummy folder/file for testing if running locally without data
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER)
        print("Creating dummy data for test...")
        conn = sqlite3.connect(os.path.join(SOURCE_FOLDER, "dummy_patient.db"))
        conn.execute("CREATE TABLE dummy_patient (patientunitstayid INTEGER, age INTEGER)")
        conn.execute("INSERT INTO dummy_patient VALUES (1, 50)")
        conn.commit()
        conn.close()

    merge_databases(SOURCE_FOLDER, OUTPUT_FILE, use_ram_disk=USE_RAM_DISK, max_workers=CORES)
