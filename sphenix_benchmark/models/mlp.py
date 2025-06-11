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
                 hidden_features_list,
                 output_features, *,
                 hidden_activ,
                 output_activ,
                 hidden_norm,
                 output_norm,
                 input_dropout,
                 hidden_dropout):

        super().__init__()

        layers = []

        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))

        # Hidden layers
        in_f = input_features
        for layer_id, out_f in enumerate(hidden_features_list):

            layers += [nn.Linear(in_f, out_f),
                       get_norm_layer(1, hidden_norm, out_f),
                       get_activ_layer(hidden_activ)]

            if hidden_dropout > 0:
                layers.append(nn.Dropout(hidden_dropout))

            in_f = out_f

        # Final layer
        layers.append(nn.Linear(in_f, output_features))
        if output_activ is not None:
            layers += [get_norm_layer(1, output_norm, output_features),
                       get_activ_layer(output_activ)]

        self.model = nn.Sequential(*layers)

    def forward(self, data):
        return self.model(data)


def test():
    """
    test input processor
    """
    with open('test_config.yaml', 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    # load data
    model_config = config['model']
    model = MLP(**model_config)

    data = torch.randn(10, model_config['input_features'])
    print(model(data))


if __name__ == '__main__':
    test()
