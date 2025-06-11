"""
Fully connect layers for hit embedding
"""
import yaml

import torch
from torch import nn

from sphenix_benchmark.utils import get_norm_layer, get_activ_layer


class MLP(nn.Module):
    """
    Fully connect layers for hit embedding
    """
    def __init__(self,
                 input_features,
                 hidden_features,
                 output_features,
                 hidden_activ,
                 output_activ,
                 norm):

        super().__init__()

        hidden_activ = get_activ_layer(hidden_activ)
        output_activ = get_activ_layer(output_activ)

        layers = []

        # Hidden layers
        in_f = input_features
        for out_f in hidden_features:
            layers += [nn.Linear(in_f, out_f),
                       get_norm_layer(1, norm, out_f),
                       hidden_activ]
            in_f = out_f

        # Final layer
        layers += [nn.Linear(in_f, output_features),
                   get_norm_layer(1, norm, output_features),
                   output_activ]

        self.model = nn.Sequential(*layers)

    def forward(self, data):
        return self.model(data)


def test():
    """
    test input processor
    """
    with open('config.yaml', 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    # load data
    filter_model_config = config['filter_model']
    model = MLP(**filter_model_config)

    data = torch.randn(10, filter_model_config['input_features'])
    model(data)


if __name__ == '__main__':
    test()
