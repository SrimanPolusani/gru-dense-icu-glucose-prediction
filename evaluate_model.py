import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report
import seaborn as sns
import json

# Import model definition from hybrid_model.py
# We assume hybrid_model.py is in the same directory
try:
    from hybrid_model import build_hybrid_model, DataGenerator
except ImportError:
    print("Error: Could not import 'hybrid_model.py'. Make sure it is in the current directory.")
    exit(1)

def evaluate_full_dataset(data_dir, weights_path='best_model.weights.h5', batch_size=1024):
    print(f"--- Starting Evaluation ---")
    print(f"Data Directory: {data_dir}")
    print(f"Weights Path: {weights_path}")

    # 1. Load Model Structure
    # Detect shapes dynamically from the first file in data_dir
    try:
        first_file = [f for f in os.listdir(data_dir) if f.endswith('.npz')][0]
        data_path = os.path.join(data_dir, first_file)
        data = np.load(data_path)
        input_shape_seq = data['X_seq'].shape[1:]
        input_shape_static = data['X_static'].shape[1:]
        print(f"Detected Sequence Shape: {input_shape_seq}")
        print(f"Detected Static Shape: {input_shape_static}")
    except Exception as e:
        print(f"Error detecting shapes from data: {e}")
        print("Falling back to default shapes (24, 96) and (37,)")
        input_shape_seq = (24, 96)
        input_shape_static = (37,)
    
    print("Building model...")
    # Use MirroredStrategy if GPUs are available, else default
    strategy = tf.distribute.MirroredStrategy()
    # Load Normalization Stats
    norm_stats = None
    # Search locations: script directory, current working directory, data directory
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normalization_stats.json'),
        os.path.join(os.getcwd(), 'normalization_stats.json'),
        os.path.join(data_dir, 'normalization_stats.json')
    ]
    
    stats_file = None
    for path in search_paths:
        if os.path.exists(path):
            stats_file = path
            break
            
    if stats_file:
        print(f"Loading normalization statistics from: {stats_file}")
        with open(stats_file, 'r') as f:
            norm_stats = json.load(f)
    else:
        print(f"WARNING: normalization_stats.json not found in {search_paths}")
        print("Proceeding with unnormalized data (matching likely training configuration).")

    with strategy.scope():
        model = build_hybrid_model(input_shape_seq, input_shape_static, norm_stats=norm_stats)
        try:
            model.load_weights(weights_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return

    # 2. Create Data Generator (Shuffle=False is CRITICAL for matching predictions to targets)
    print("Initializing Data Generator...")
    gen = DataGenerator(data_dir, batch_size=batch_size, shuffle=False, test_mode=False)
    
    # 3. Run Predictions
    print("Running predictions on full dataset (this may take a while)...")
    # We use the generator for prediction
    predictions = model.predict(gen, verbose=1)
    y_pred_hyper = predictions[0].flatten()
    y_pred_hypo = predictions[1].flatten()
    
    # 4. Collect Ground Truth
    # We must iterate the generator exactly as predict() did to get matching labels
    print("Collecting ground truth labels...")
    y_true_hyper = []
    y_true_hypo = []
    
    # Note: gen has __len__ so we can loop range(len(gen))
    for i in range(len(gen)):
        if i % 100 == 0:
            print(f"Processing batch {i}/{len(gen)}")
        _, (y_hyper_batch, y_hypo_batch) = gen[i]
        y_true_hyper.append(y_hyper_batch)
        y_true_hypo.append(y_hypo_batch)
        
    y_true_hyper = np.concatenate(y_true_hyper).flatten()
    y_true_hypo = np.concatenate(y_true_hypo).flatten()
    
    # Truncate predictions if necessary (generator might drop last partial batch depending on implementation)
    # But our DataGenerator drops partials, so lengths should match.
    min_len = min(len(y_pred_hyper), len(y_true_hyper))
    y_pred_hyper = y_pred_hyper[:min_len]
    y_pred_hypo = y_pred_hypo[:min_len]
    y_true_hyper = y_true_hyper[:min_len]
    y_true_hypo = y_true_hypo[:min_len]
    
    print(f"Total Samples Evaluated: {min_len}")

    # 5. Create Results DataFrame
    df = pd.DataFrame({
        'Target_Hyper': y_true_hyper,
        'Pred_Hyper': y_pred_hyper,
        'Target_Hypo': y_true_hypo,
        'Pred_Hypo': y_pred_hypo
    })
    
    # 6. Analysis & Visualization
    
    # Function to find optimal threshold (Maximize F1)
    def find_optimal_threshold(y_true, y_pred):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], f1_scores[best_idx]

    thresh_hyper, f1_hyper = find_optimal_threshold(y_true_hyper, y_pred_hyper)
    thresh_hypo, f1_hypo = find_optimal_threshold(y_true_hypo, y_pred_hypo)
    
    print(f"\n--- Optimal Thresholds (Max F1) ---")
    print(f"Hypertension: Threshold={thresh_hyper:.4f}, F1={f1_hyper:.4f}")
    print(f"Hypotension:  Threshold={thresh_hypo:.4f}, F1={f1_hypo:.4f}")
    
    # Apply Thresholds
    df['Pred_Hyper_Class'] = (df['Pred_Hyper'] > thresh_hyper).astype(int)
    df['Pred_Hypo_Class'] = (df['Pred_Hypo'] > thresh_hypo).astype(int)
    
    # Save to CSV
    csv_path = 'evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to {csv_path}")
    
    # Generate HTML Report
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h2 {{ color: #333; }}
            .metric-box {{ background: #f9f9f9; padding: 15px; border: 1px solid #ccc; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        
        <div class="metric-box">
            <h2>Hypertension (Hyper) Performance</h2>
            <p><strong>AUC:</strong> {roc_auc_score(y_true_hyper, y_pred_hyper):.4f}</p>
            <p><strong>Optimal Threshold:</strong> {thresh_hyper:.4f}</p>
            <pre>{classification_report(y_true_hyper, df['Pred_Hyper_Class'])}</pre>
        </div>

        <div class="metric-box">
            <h2>Hypotension (Hypo) Performance</h2>
            <p><strong>AUC:</strong> {roc_auc_score(y_true_hypo, y_pred_hypo):.4f}</p>
            <p><strong>Optimal Threshold:</strong> {thresh_hypo:.4f}</p>
            <pre>{classification_report(y_true_hypo, df['Pred_Hypo_Class'])}</pre>
        </div>

        <h2>Sample Predictions (Top 50)</h2>
        {df.head(50).to_html(classes='table', index=False, float_format=lambda x: '%.4f' % x)}
        
    </body>
    </html>
    """
    
    with open('evaluation_report.html', 'w') as f:
        f.write(html_content)
    print("Report saved to evaluation_report.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed_tensors')
    parser.add_argument('--weights', type=str, default='best_model.weights.h5')
    # If running in a notebook, parse_args() fails because Jupyter passes extra args.
    # We use parse_known_args() or default to empty list if in a notebook.
    try:
        args = parser.parse_args()
    except SystemExit:
        # This happens in Jupyter. Use defaults.
        print("Detected Jupyter Environment. Using default arguments.")
        args = parser.parse_args([])
    
    evaluate_full_dataset(args.data_dir, args.weights)
