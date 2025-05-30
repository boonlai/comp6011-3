import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def grad_cam(
    model, input_tensor, target_class, output_path, *, target_name=None, silent=False
):
    """
    Generate GRAD-CAM visualization for the DINOv2 backbone.
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the last norm layer of the backbone
    target_layer = model.backbone.blocks[-1].norm1
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    try:
        # Forward and backward passes
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        class_score = output[0, target_class]
        model.zero_grad()
        class_score.backward(retain_graph=True)

        # Get gradients and activations
        gradients_tensor = gradients[0]  # ViT-S: (1, 197, 384)
        activations_tensor = activations[0]  # ViT-S: (1, 197, 384)

        # Calculate attention weights
        pooled_gradients = torch.mean(gradients_tensor, dim=[0, 2])
        for i in range(activations_tensor.shape[1]):
            activations_tensor[0, i, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = torch.mean(activations_tensor, dim=2).squeeze()
        heatmap = heatmap[1:]  # Remove CLS token
        patch_size = int(np.sqrt(len(heatmap)))
        heatmap = heatmap.reshape(patch_size, patch_size)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.detach().cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (518, 518))

        # Prepare input image for visualization
        input_np = input_tensor.squeeze().detach().cpu().numpy()
        input_np = np.transpose(input_np, (1, 2, 0))
        input_np = (input_np * 0.5) + 0.5
        input_np = np.clip(input_np, 0, 1)
        if input_np.shape[2] == 3:
            input_np = np.mean(input_np, axis=2)

        # Create visualization
        plt.figure(figsize=(5, 5))
        plt.imshow(input_np, cmap="gray", alpha=0.7)
        plt.imshow(heatmap_resized, cmap="jet", alpha=0.3)
        plt.title(
            "Target Prediction" if target_name is None else f"Target: {target_name}"
        )
        plt.axis("off")

        # ECG lead labels in the same (standard) order as the export
        # Verified with `meta.get("sig_names")`
        lead_names = [
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        rows, cols = 4, 3
        for i in range(len(lead_names)):
            row = i // cols
            col = i % cols
            x = col / cols + 0.15
            y = row / rows + 0.15
            plt.text(
                x * 518,
                y * 518,
                lead_names[i],
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            )

        # Save and cleanup
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        if not silent:
            print(f"GRAD-CAM visualization saved to: {output_path}")

        return heatmap_resized

    finally:
        handle_forward.remove()
        handle_backward.remove()
