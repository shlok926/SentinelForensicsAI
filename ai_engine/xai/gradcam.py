import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from typing import Tuple

from ai_engine.fusion.late_fusion import LateFusionClassifier

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) visualizer.
    Highlights facial regions targeted by the model during classification.
    """
    def __init__(self, model: LateFusionClassifier, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        # Hooks to capture gradients and feature maps
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)

    def generate_heatmap(
        self, 
        face_tensor: torch.Tensor, 
        mel_tensor: torch.Tensor,
        original_image_path: str,
        output_path: str
    ) -> str:
        """
        Computes Grad-CAM heatmap, overlays it on the face image, and saves to disk.
        """
        self.model.eval()
        self.model.zero_grad()

        # 1. Forward pass
        logits = self.model(face_tensor, mel_tensor)
        
        # 2. Backward pass for target logit score
        logits.backward()

        # 3. Retrieve gradients and feature activations
        # shape of gradients: [batch, channels, H, W]
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # 4. Compute channel-wise weights (Global Average Pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2)) # [channels]

        # 5. Compute weighted combination of activation channels
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 6. Apply ReLU (we only care about features that positively contribute to class decision)
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # 7. Load original image and resize heatmap
        orig_img = cv2.imread(original_image_path)
        if orig_img is None:
            # Fallback if original image cannot be loaded
            orig_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        h, w, _ = orig_img.shape
        heatmap = cv2.resize(cam, (w, h))

        # 8. Apply colormap (Jet) and overlay
        heatmap = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on original frame (60% original, 40% heatmap)
        overlay = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)

        # Ensure output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)

        return output_path
