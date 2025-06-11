import torch
from torch import nn
import torch.nn.functional as F


class ExaTrkXEmbedder(nn.Module):

    """
    We set parameters from the Exa.TrkX paper as default
    NOTE: The yaml file in the repo "train_quickstart_embedding.yaml"
    has different values for several parameters.
    When there are contradictions, we use the paper parameter.
    """

    def __init__(self,
                 input_size,
                 # NOTE: there is a conflict in the output dimension
                 # section 5.2 said it is 8 in paragraph 1, but the
                 # last paragraph (and the yaml in the repo) indicated
                 # it should be 12. Let use 12 here since it is bigger.
                 sizes                = [1024] * 4 + [12],
                 hidden_activation    = 'Tanh',
                 output_activation    = None,
                 layer_norm           = True,
                 output_normalization = True):

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
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(hidden_activation())

        # Final layer
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if output_activation is not None:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[-1]))
            layers.append(output_activation())

        self.layers = nn.Sequential(*layers)

        self.output_normalization = output_normalization

    def forward(self, x):

        x_out = self.layers(x)

        if self.output_normalization:
            return F.normalize(x_out)

        return x_out


class HingeLoss(nn.Module):
    """
    According to the Exa.TrkX paper, we use p=2 and margin = 1
    The the distance is NOT averaged by the number of features.
    """
    def __init__(self, p=2., margin=1.):
        super().__init__()
        self.loss_fn = nn.HingeEmbeddingLoss(margin=margin)
        self.p = p

    def forward(self, head, tail, labels, return_dist=False):
        dist = torch.pow(head - tail, self.p).sum(-1)
        loss = self.loss_fn(input=dist, target=2. * labels - 1)

        if return_dist:
            return loss, dist
        return dist