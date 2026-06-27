import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import numpy as np

from ai_engine.fusion.late_fusion import LateFusionClassifier
from ai_engine.xai.gradcam import GradCAM

def main():
    print("Generating Explainable AI (XAI) Grad-CAM report...")

    model_path = "models/multimodal_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please complete training first.")
        return

    # 1. Load sample dataset index
    index_csv = "storage/datasets/deepfake_detection_processed_index.csv"
    if not os.path.exists(index_csv):
        print(f"Error: Processed index CSV missing at {index_csv}")
        return

    df = pd.read_csv(index_csv)
    if df.empty:
        print("Error: Index CSV is empty.")
        return

    # Select a fake sample if possible for visual demonstration of forgery indicators
    fake_samples = df[df["label"] == 1]
    sample_row = fake_samples.iloc[0] if not fake_samples.empty else df.iloc[0]

    face_path = sample_row["face_path"]
    audio_mel_path = sample_row["audio_mel_path"]

    print(f"Selected sample for Grad-CAM: {face_path}")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LateFusionClassifier(pretrained_video=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # We target the last residual block of ResNet18
    target_layer = model.video_extractor.resnet.layer4[-1]

    # 3. Instantiate GradCAM
    gradcam = GradCAM(model, target_layer)

    # 4. Prepare tensors
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(face_path).convert("RGB")
    face_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, 224, 224]

    mel_array = np.load(audio_mel_path)
    mel_tensor = torch.tensor(mel_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 128, T]

    # 5. Generate and save heatmap overlay
    output_path = "results/xai_gradcam_overlay.png"
    gradcam.generate_heatmap(
        face_tensor=face_tensor,
        mel_tensor=mel_tensor,
        original_image_path=face_path,
        output_path=output_path
    )

    print(f"Explainable AI heatmap saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
