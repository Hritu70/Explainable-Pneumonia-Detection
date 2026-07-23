
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):

        self.model.zero_grad()

        output = self.model(input_tensor)

        predicted_class = output.argmax(dim=1)

        output[:, predicted_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


def create_overlay(original, heatmap):

    original = np.array(original) / 255.0

    colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]

    overlay = (
        0.6 * original +
        0.4 * colored_heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    return colored_heatmap, overlay
