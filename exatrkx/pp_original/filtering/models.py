import torch
from torch import nn


class ExaTrkXFilter(nn.Module):

    """
    We set parameters from the Exa.TrkX paper as default
    NOTE: The yaml file in the repo "train_quickstart_embedding.yaml"
    has different values for several parameters.
    When there are contradictions, we use the paper parameter.
    """

    def __init__(self,
                 input_size        = 4 * 2,
                 sizes             = [512] * 3 + [1],
                 hidden_activation = 'Tanh',
                 output_activation = None,
                 layer_norm        = True):

        super().__init__()

        hidden_activation = getattr(nn, hidden_activation)

        if output_activation is not None:
            output_activation = getattr(nn, output_activation)

        layers = []
        n_layers = len(sizes)
        sizes = [input_size] + sizes

        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(hidden_activation())
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))

        # Final layer
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, head, tail):
        return self.layers(torch.cat([head, tail], dim=-1))
