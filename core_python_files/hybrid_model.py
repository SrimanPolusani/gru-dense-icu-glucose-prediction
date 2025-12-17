import os
import sys
import logging

# --- AUTO-CONFIGURE CPU OPTIMIZATIONS ---
# Must be done before importing TensorFlow to ensure OMP/KMP vars take effect.
# This allows the script to run optimally with ANY Slurm submission script.

# Check if we are in a Slurm environment and have CPU constraints
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if slurm_cpus:
    num_threads = slurm_cpus
    print(f"Auto-Config: Detected SLURM_CPUS_PER_TASK={num_threads}. Configuring for high-performance CPU training.")
else:
    # Default to 32 as requested by user if not specified
    num_threads = '32'
    print(f"Auto-Config: SLURM_CPUS_PER_TASK not found. Defaulting to {num_threads} threads for optimization.")

# Set Environment Variables for Intel/CPU Performance
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['TF_NUM_INTRAOP_THREADS'] = num_threads
os.environ['TF_NUM_INTEROP_THREADS'] = '2' # 2 is usually sufficient for inter-op
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_SETTINGS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

# Suppress TensorFlow CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
import argparse
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import json
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Apply TF Threading (Redundant if env vars work, but good for safety)
tf.config.threading.set_intra_op_parallelism_threads(int(num_threads))
tf.config.threading.set_inter_op_parallelism_threads(2)
logging.info(f"CPU Optimization: Intra-op threads set to {num_threads}")

class FocalLoss(tf.keras.losses.Loss):
    """
    Implements Focal Loss: -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.75, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate pt (probability of true class)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha factor
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Calculate focal weight
        focal_weight = alpha_factor * tf.pow(1.0 - pt, self.gamma)
        
        # Calculate Cross Entropy
        ce = -tf.math.log(pt)
        
        return focal_weight * ce

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras model from .npz files using File-Based Shuffling.
    Loads files one by one to minimize RAM usage and disk thrashing.
    """
    def __init__(self, data_dir, batch_size=32, shuffle=True, test_mode=False, num_workers=1, worker_id=0, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test_mode = test_mode
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.file_list = []
        self.batch_map = [] # List of (filename, start_index, end_index)
        self.cache = {} # Holds the current file data
        self.current_file_path = None
        
        if not self.test_mode:
            if os.path.exists(data_dir):
                all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
                all_files.sort()
                # Shard files
                self.file_list = [f for i, f in enumerate(all_files) if i % self.num_workers == self.worker_id]
                logging.info(f"Worker {self.worker_id}: Found {len(self.file_list)} files.")
                
                # Initial scan to validate files (optional but good for safety)
                # We won't build a global index here, we'll do it in on_epoch_end
            else:
                logging.warning(f"Data directory {data_dir} not found.")
        else:
            logging.info("Running in TEST mode with synthetic data.")
            self.file_list = ['synthetic']
            
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.batch_map)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        
        if self.test_mode:
            for i in range(10):
                self.batch_map.append(('synthetic', 0, self.batch_size))
            return

        # 1. Shuffle the ORDER of files
        current_files = self.file_list.copy()
        if self.shuffle:
            np.random.shuffle(current_files)
            
        # 2. Build the batch map based on this new order
        # This ensures we process File A -> File B -> File C sequentially
        logging.info(f"Worker {self.worker_id}: Rebuilding batch map for new epoch...")
        
        new_batch_map = []
        for f in current_files:
            try:
                # We need the sample count. 
                # Optimization: Cache sample counts to avoid re-opening files just to check size?
                # For now, let's just open it. It's 1340 opens per epoch, which is negligible (0.5s total).
                file_path = os.path.join(self.data_dir, f)
                
                # Fast peek at shape
                try:
                    with np.load(file_path, mmap_mode='r') as data:
                         n_samples = data['y_hyper'].shape[0]
                except:
                    # Fallback
                    data = np.load(file_path)
                    n_samples = data['y_hyper'].shape[0]
                    del data
                
                # Create batches for this file
                # We drop the last partial batch to keep things aligned and simple
                # or we can yield it. Let's yield it if it's non-empty.
                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    if end > start:
                        new_batch_map.append((f, start, end))
                        
            except Exception as e:
                logging.error(f"Error scanning {f}: {e}")
        
        self.batch_map = new_batch_map
        logging.info(f"Worker {self.worker_id}: Epoch prepared. {len(self.batch_map)} batches queued.")

    def _get_data_from_file(self, file_path):
        """Loads a file with caching."""
        # If we are already holding this file, return it
        if self.current_file_path == file_path and self.cache:
            return self.cache
            
        # Otherwise load new file
        try:
            # logging.info(f"Loading file: {os.path.basename(file_path)}")
            data = np.load(file_path)
            # Convert to dict to keep in memory
            self.cache = {
                'X_seq': data['X_seq'],
                'X_static': data['X_static'],
                'y_hyper': data['y_hyper'],
                'y_hypo': data['y_hypo']
            }
            self.current_file_path = file_path
            return self.cache
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.test_mode:
            return self.__data_generation_synthetic()

        # Get batch info
        filename, start, end = self.batch_map[index]
        file_path = os.path.join(self.data_dir, filename)
        
        # Load data (cached if same file as previous call)
        data = self._get_data_from_file(file_path)
        
        if data:
            # Extract batch
            X_seq = data['X_seq'][start:end]
            X_static = data['X_static'][start:end]
            y_hyper = data['y_hyper'][start:end]
            y_hypo = data['y_hypo'][start:end]
            
            # --- DYNAMIC UNDERSAMPLING (Enriched Dataset) ---
            # Only apply during TRAINING (not validation/test)
            if self.shuffle:
                # Identify indices
                idx_pos = np.where((y_hyper == 1) | (y_hypo == 1))[0]
                idx_neg = np.where((y_hyper == 0) & (y_hypo == 0))[0]
                
                # Keep ALL positives
                # Keep 10% of negatives (Enrichment)
                if len(idx_neg) > 0:
                    keep_neg_count = max(1, int(len(idx_neg) * 0.1)) # 10% retention
                    idx_neg_keep = np.random.choice(idx_neg, size=keep_neg_count, replace=False)
                else:
                    idx_neg_keep = np.array([], dtype=int)
                
                # Combine and Shuffle
                final_indices = np.concatenate([idx_pos, idx_neg_keep])
                np.random.shuffle(final_indices)
                
                # Apply mask
                X_seq = X_seq[final_indices]
                X_static = X_static[final_indices]
                y_hyper = y_hyper[final_indices]
                y_hypo = y_hypo[final_indices]
            
            return {
                'input_seq': X_seq, 
                'input_static': X_static
            }, (y_hyper, y_hypo)
        else:
            return None

    def __data_generation_synthetic(self):
        # Generate random data for testing
        X_seq = np.random.rand(self.batch_size, 24, 96).astype(np.float32)
        X_static = np.random.rand(self.batch_size, 37).astype(np.float32)
        
        y_hyper = np.random.randint(0, 2, self.batch_size).astype(np.float32)
        y_hypo = np.random.randint(0, 2, self.batch_size).astype(np.float32)
        
        return {'input_seq': X_seq, 'input_static': X_static}, (y_hyper, y_hypo)

# --- Custom Layers ---
class TemporalAttention(layers.Layer):
    """
    Computes a weighted average of the time steps (Attention).
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # x: (batch, time, features)
        # e: (batch, time, 1)
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        # a: (batch, time, 1)
        a = keras.backend.softmax(e, axis=1)
        # output: (batch, features)
        output = x * a
        return keras.backend.sum(output, axis=1)

def build_hybrid_model(input_shape_seq, input_shape_static, norm_stats=None):
    """
    Builds the Hybrid GRU-Dense model with Input Normalization.
    """
    # --- Temporal Branch ---
    input_seq = layers.Input(shape=input_shape_seq, name='input_seq')
    
    # 1. Normalization Layer (CRITICAL FIX)
    # 1. Normalization Layer (CRITICAL FIX)
    if norm_stats:
        logging.info("Initializing Normalization layer with pre-computed stats.")
        # Mean and Variance from compute_stats.py
        mean = np.array(norm_stats['mean'], dtype=np.float32)
        variance = np.array(norm_stats['variance'], dtype=np.float32)
        std = np.sqrt(variance)
        
        # Use Lambda layer for compatibility with older TF versions
        # (x - mean) / std
        norm_layer = layers.Lambda(lambda x: (x - mean) / (std + 1e-7))
        x_seq = norm_layer(input_seq)
    else:
        logging.warning("No normalization stats provided. Using raw inputs (NOT RECOMMENDED).")
        x_seq = input_seq

    x_seq = layers.Masking(mask_value=0.0)(x_seq)
    x_seq = layers.GRU(64, return_sequences=True)(x_seq)
    x_seq = layers.Dropout(0.3)(x_seq)
    x_seq = layers.GRU(32, return_sequences=True)(x_seq) # Changed to True for Attention
    x_seq = layers.Dropout(0.3)(x_seq)
    
    # Attention Layer
    x_seq = TemporalAttention()(x_seq)
    
    # --- Static Branch ---
    input_static = layers.Input(shape=input_shape_static, name='input_static')
    x_static = layers.Dense(32, activation='relu')(input_static)
    x_static = layers.Dropout(0.2)(x_static)
    
    # --- Concatenate ---
    combined = layers.Concatenate()([x_seq, x_static])
    
    # --- Shared Dense Layers ---
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # --- Output Heads ---
    output_hyper = layers.Dense(1, activation='sigmoid', name='output_hyper')(x)
    output_hypo = layers.Dense(1, activation='sigmoid', name='output_hypo')(x)
    
    model = models.Model(inputs=[input_seq, input_static], outputs=[output_hyper, output_hypo])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'output_hyper': FocalLoss(alpha=0.75, gamma=2.0),
            'output_hypo': FocalLoss(alpha=0.75, gamma=2.0)
        },
        loss_weights={'output_hyper': 1.0, 'output_hypo': 1.0}, 
        metrics={
            'output_hyper': [keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')],
            'output_hypo': [keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        }
    )
    
    return model

def train_model(model, train_gen, val_gen, epochs=10):
    """
    Trains the model with callbacks.
    """
    checkpoint_dir = os.path.dirname(train_gen.data_dir) if train_gen.data_dir else '.'
    best_weights_path = 'best_model.weights.h5'
    latest_weights_path = 'latest_model.weights.h5'
    
    # Check for existing checkpoints to resume
    initial_epoch = 0
    if os.path.exists(latest_weights_path):
        logging.info(f"Found checkpoint '{latest_weights_path}'. Loading weights to resume training...")
        try:
            model.load_weights(latest_weights_path)
            logging.info("Weights loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(best_weights_path, save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1),
        keras.callbacks.ModelCheckpoint(latest_weights_path, save_best_only=False, save_weights_only=True, verbose=1)
    ]
    
    # IMPORTANT: shuffle=False is required for File-Based Shuffling to work efficiently.
    # We handle shuffling manually in DataGenerator.on_epoch_end by shuffling the file order.
    # If shuffle=True (default), Keras will request batches in random order, causing disk thrashing.
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        shuffle=False, # <--- CRITICAL: Disable Keras global shuffling
    )
    return history

def setup_slurm_cluster():
    """
    Sets up the TF_CONFIG environment variable for multi-node training on Slurm.
    Returns: (is_slurm, num_workers, worker_id)
    """
    slurm_job_nodelist = os.environ.get('SLURM_JOB_NODELIST')
    if not slurm_job_nodelist:
        logging.warning("No SLURM_JOB_NODELIST found. Assuming single-node execution.")
        return False, 1, 0

    # Parse nodelist (e.g., "node[01-02]" or "node01,node02")
    try:
        import subprocess
        result = subprocess.run(['scontrol', 'show', 'hostnames', slurm_job_nodelist], 
                              capture_output=True, text=True)
        nodes = result.stdout.strip().split('\n')
    except Exception as e:
        logging.error(f"Failed to parse nodelist with scontrol: {e}")
        return False, 1, 0

    if not nodes:
        return False, 1, 0

    # Get current node rank
    try:
        # Check SLURM_NNODES
        n_nodes = int(os.environ.get('SLURM_NNODES'))
        node_id = int(os.environ.get('SLURM_NODEID'))
    except (ValueError, TypeError):
        logging.warning("Could not get Slurm rank/node info.")
        return False, 1, 0

    # Construct cluster spec
    # Construct cluster spec
    if n_nodes > 1:
        cluster_spec = {
            "cluster": {
                "worker": [f"{node}:12345" for node in nodes]
            },
            "task": {
                "type": "worker",
                "index": node_id
            }
        }

        os.environ['TF_CONFIG'] = json.dumps(cluster_spec)
        logging.info(f"TF_CONFIG set for node {node_id}/{n_nodes}: {cluster_spec}")
    else:
        logging.info(f"Single node Slurm job (Node {node_id}/{n_nodes}). Skipping TF_CONFIG for distributed training.")
    return True, n_nodes, node_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid GRU-Dense Model")
    parser.add_argument('--test', action='store_true', help="Run in test mode with synthetic data")
    parser.add_argument('--data_dir', type=str, default='processed_tensors', help="Directory containing .npz files")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per worker")
    
    args = parser.parse_args()

    # Setup Slurm Cluster for Multi-Node
    is_slurm, num_workers, worker_id = setup_slurm_cluster()

    # Define Strategy
    if is_slurm and num_workers > 1:
        print(f"--- RUNNING IN MULTI-NODE MODE (Worker {worker_id}/{num_workers}) ---")
        # For CPU multi-worker, MultiWorkerMirroredStrategy is still the correct choice.
        # It will detect available devices (CPUs) if no GPUs are visible.
        try:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        except AttributeError:
            logging.warning("tf.distribute.MultiWorkerMirroredStrategy not found. Trying experimental version.")
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        print("--- RUNNING IN SINGLE-NODE MODE (MirroredStrategy) ---")
        strategy = tf.distribute.MirroredStrategy()
        
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    if args.test:
        print("--- RUNNING IN TEST MODE ---")
        # Synthetic shapes
        input_shape_seq = (24, 96)
        input_shape_static = (37,)
        
        with strategy.scope():
            model = build_hybrid_model(input_shape_seq, input_shape_static)
        model.summary()
        
        train_gen = DataGenerator(args.data_dir, test_mode=True, batch_size=args.batch_size, num_workers=num_workers, worker_id=worker_id)
        val_gen = DataGenerator(args.data_dir, test_mode=True, batch_size=args.batch_size, num_workers=num_workers, worker_id=worker_id)
        
        print("Starting training loop...")
        train_model(model, train_gen, val_gen, epochs=1)
        print("Test run complete.")
        
    else:
        print(f"--- RUNNING WITH DATA FROM {args.data_dir} ---")
        # Need to determine shapes from data or hardcode if known
        # For now, let's try to load one file to get shapes
        if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
            logging.error(f"No data found in {args.data_dir}. Run with --test to verify architecture.")
            exit(1)
            
        first_file = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')][0]
        data = np.load(os.path.join(args.data_dir, first_file))
        input_shape_seq = data['X_seq'].shape[1:]
        input_shape_static = data['X_static'].shape[1:]
        
        print(f"Detected Sequence Shape: {input_shape_seq}")
        print(f"Detected Static Shape: {input_shape_static}")
        
        # Load Normalization Stats
        norm_stats = None
        stats_file = 'normalization_stats.json'
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                norm_stats = json.load(f)
            print("Loaded normalization statistics.")
        else:
            print("WARNING: normalization_stats.json not found. Model will train on unscaled data.")
        
        with strategy.scope():
            model = build_hybrid_model(input_shape_seq, input_shape_static, norm_stats=norm_stats)
        model.summary()
        
        train_gen = DataGenerator(args.data_dir, test_mode=False, batch_size=args.batch_size, num_workers=num_workers, worker_id=worker_id)
        # In real scenario, split files into train/val
        # For now, using same for both just to demonstrate flow if files exist
        val_gen = DataGenerator(args.data_dir, test_mode=False, batch_size=args.batch_size, num_workers=num_workers, worker_id=worker_id) 
        
        train_model(model, train_gen, val_gen, epochs=args.epochs)
