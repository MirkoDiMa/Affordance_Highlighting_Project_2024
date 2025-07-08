# src/models/neural_highlighter.py

import torch
import torch.nn as nn

class NeuralHighlighter(nn.Module):
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_classes: int = 2):
        """
        MLP mapping 3D coordinate → [prob_not_region, prob_region].
        Follows: 6 Linear layers, ReLU+LayerNorm after first 5, Softmax at end. :contentReference[oaicite:1]{index=1}
        """
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,3] vertex coordinates
        returns: [N,2] probabilities per vertex
        """
        logits = self.mlp(x)
        probs = self.softmax(logits)
        return probs
