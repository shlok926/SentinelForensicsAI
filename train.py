import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ai_engine.training.dataset import MultimodalForensicsDataset
from ai_engine.fusion.late_fusion import LateFusionClassifier

def train_model(
    processed_csv: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.0005,
    device: str = "cpu"
) -> None:
    """
    Core PyTorch training runner.
    Trains the LateFusionClassifier using BCEWithLogitsLoss.
    Saves the best model weights.
    """
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(processed_csv):
        raise FileNotFoundError(
            f"Processed index CSV not found: {processed_csv}. "
            "Please run: python -m scripts.run_bulk_preprocessing first."
        )

    # 1. Load and Split CSV
    df = pd.read_csv(processed_csv)
    print(f"Loaded {len(df)} samples from processed index.")

    # Stratified split to preserve class distribution
    try:
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df["label"], 
            random_state=42
        )
    except ValueError:
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42
        )
    print(f"Split completed: Train samples={len(train_df)}, Val samples={len(val_df)}")

    # Temporary CSVs for PyTorch loaders
    train_csv_path = "storage/datasets/train_processed_temp.csv"
    val_csv_path = "storage/datasets/val_processed_temp.csv"
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # 2. Instantiate PyTorch Datasets and Dataloaders
    train_dataset = MultimodalForensicsDataset(train_csv_path, split="train")
    val_dataset = MultimodalForensicsDataset(val_csv_path, split="val")

    # Use num_workers=0 to avoid multiprocessing issues on standard desktop platforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate Late Fusion Model
    print(f"Initializing Multimodal Late Fusion Classifier on device: {device}...")
    model = LateFusionClassifier(pretrained_video=False)
    model.to(device)

    # 4. Optimizer, Scheduler & Loss
    # We use BCEWithLogitsLoss for numerical stability (combines Sigmoid layer + BCELoss)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    print("Beginning training loop...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        for faces, mels, labels in train_loader:
            faces = faces.to(device)
            mels = mels.to(device)
            labels = labels.to(device).unsqueeze(1) # reshape: [B, 1]

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(faces, mels)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item() * faces.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train_samples += faces.size(0)

        epoch_train_loss = train_loss / total_train_samples if total_train_samples > 0 else 0
        epoch_train_acc = train_correct / total_train_samples if total_train_samples > 0 else 0

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for faces, mels, labels in val_loader:
                faces = faces.to(device)
                mels = mels.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(faces, mels)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * faces.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val_samples += faces.size(0)

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = val_correct / total_val_samples if total_val_samples > 0 else 0
        
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}"
        )

        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            save_path = "models/multimodal_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model weights: {save_path} (Val Loss: {epoch_val_loss:.4f})")

    # Cleanup temp split files
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)

    print("\nTraining completed successfully!")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Late Fusion Deepfake Model")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="storage/datasets/deepfake_detection_processed_index.csv", 
        help="Path to processed features index CSV"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(
        processed_csv=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device.type
    )
