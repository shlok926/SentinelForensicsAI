import os
import torch
import cv2
import numpy as np
import pytest
from PIL import Image

from ai_engine.fusion.late_fusion import LateFusionClassifier
from ai_engine.xai.gradcam import GradCAM

def test_gradcam_generation(tmp_path):
    # 1. Instantiate late fusion model
    model = LateFusionClassifier(pretrained_video=False)
    
    # 2. Select the target layer (last residual block of ResNet18)
    target_layer = model.video_extractor.resnet.layer4[-1]
    
    # 3. Instantiate GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # 4. Generate dummy inputs
    face_tensor = torch.randn((1, 3, 224, 224))
    mel_tensor = torch.randn((1, 1, 128, 300))
    
    # Create dummy original image file
    orig_img_path = tmp_path / "dummy_face.jpg"
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(orig_img_path), dummy_img)
    
    output_heatmap_path = tmp_path / "gradcam_output.png"
    
    # 5. Execute Grad-CAM
    res_path = gradcam.generate_heatmap(
        face_tensor=face_tensor,
        mel_tensor=mel_tensor,
        original_image_path=str(orig_img_path),
        output_path=str(output_heatmap_path)
    )
    
    # 6. Verifications
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 0
    
    # Verify the generated image can be loaded and has correct shape
    loaded_img = cv2.imread(res_path)
    assert loaded_img is not None
    assert loaded_img.shape == (224, 224, 3)
