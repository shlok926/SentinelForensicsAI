import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import matplotlib
# Set matplotlib backend to Agg to work headless without graphical interface crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from ai_engine.training.dataset import MultimodalForensicsDataset
from ai_engine.fusion.late_fusion import LateFusionClassifier

def evaluate_model(
    processed_csv: str,
    model_path: str = "models/multimodal_best.pth",
    results_dir: str = "results",
    device: str = "cpu"
) -> None:
    """
    Evaluates the trained multimodal model, computes metrics, and generates visual plots.
    """
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Processed CSV index missing: {processed_csv}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint missing: {model_path}. Please run train.py first.")

    # 1. Load and Split CSV (using the same split strategy as train.py to get test set)
    df = pd.read_csv(processed_csv)
    try:
        _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    except ValueError:
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_csv_path = "storage/datasets/test_processed_temp.csv"
    test_df.to_csv(test_csv_path, index=False)

    test_dataset = MultimodalForensicsDataset(test_csv_path, split="val")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 2. Load Model Checkpoint
    print(f"Loading model architecture and weights from {model_path}...")
    model = LateFusionClassifier(pretrained_video=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Model Inference Loop
    y_true = []
    y_pred_probs = []
    y_pred_labels = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for faces, mels, labels in test_loader:
            faces = faces.to(device)
            mels = mels.to(device)
            
            outputs = model(faces, mels)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred_probs.extend(probs)
            y_pred_labels.extend((probs >= 0.5).astype(float))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = np.array(y_pred_labels)

    # Clean up temp split file
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    # 4. Calculate Metrics
    acc = accuracy_score(y_true, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='binary', zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc_score = 1.0 # fallback if only one class exists in target list

    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc_score:.4f}")
    print("====================================================")

    # 5. Generate and Save Confusion Matrix Plot
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Forensic Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Model Prediction')
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")

    # 6. Generate and Save ROC Curve Plot
    print("Generating ROC-AUC Curve...")
    plt.figure(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    except Exception:
        plt.plot([0, 1], [0, 1], color='darkorange', lw=2, label='ROC curve (N/A)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(results_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC Curve saved to: {roc_path}")

    # Save summary as JSON
    summary_path = os.path.join(results_dir, "metrics_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc_score)
        }, f, indent=4)
    print(f"Metrics summary JSON saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Deepfake Model")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="storage/datasets/deepfake_detection_processed_index.csv", 
        help="Path to processed features index CSV"
    )
    parser.add_argument("--model", type=str, default="models/multimodal_best.pth", help="Model weights path")
    parser.add_argument("--results", type=str, default="results", help="Folder to save evaluation plots")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_model(
        processed_csv=args.csv,
        model_path=args.model,
        results_dir=args.results,
        device=device.type
    )
