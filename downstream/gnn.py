"""
"""
import yaml
import torch
from torch import nn

import torch_geometric.nn as pyg_nn
from torch_scatter import scatter_add, scatter_mean, scatter_max

from sphenix_benchmark.models.mlp import MLP
from sphenix_benchmark.utils import get_activ_layer


class UniformGNN(nn.Module):
    def __init__(self,
                 encoder,
                 num_layers,
                 hidden_channels,
                 mp_cls,
                 predictor,
                 activ=None):

        super().__init__()

        # Node encoder
        self.encoder = encoder

        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mp_block = mp_cls(in_channels  = hidden_channels,
                              out_channels = hidden_channels)
            self.convs.append(mp_block)
        self.activ = get_activ_layer(activ)

        # Node prediction network
        self.predictor = predictor

    def forward(self, nodes, edge_index):

        # Encode node features
        nodes = self.encoder(nodes)

        # Message passing
        for conv in self.convs:
            nodes = conv(nodes, edge_index)
            nodes = self.activ(nodes)

        # Predict node class (or edge, as needed)
        return self.predictor(nodes)



def construct_uniform_gnn(model_config):

    encoder_config   = model_config['encoder']
    gnn_config       = model_config['gnn']
    predictor_config = model_config['predictor']

    # make sure the configurations are coherent
    assert gnn_config['hidden_channels'] == encoder_config['output_features']
    assert predictor_config['input_features'] == gnn_config['hidden_channels']

    # construct encoder
    encoder = MLP(**encoder_config)

    # get message-passing class
    mp_cls = getattr(pyg_nn, gnn_config['layer_name'])

    # construct predictor
    predictor = MLP(**predictor_config)

    return UniformGNN(encoder         = encoder,
                      num_layers      = gnn_config['num_layers'],
                      hidden_channels = gnn_config['hidden_channels'],
                      mp_cls          = mp_cls,
                      predictor       = predictor,
                      activ           = gnn_config['activ'])


class InteractionGNN(nn.Module):
    def __init__(self, *,
                 node_encoder,
                 edge_encoder,
                 node_network,
                 edge_network,
                 pred_network,
                 aggregation,
                 num_iterations):

        super().__init__()

        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.node_network = node_network
        self.edge_network = edge_network
        self.pred_network = pred_network

        self.aggregation = aggregation
        self.num_iterations = num_iterations

    def forward(self, nodes, edge_index):
        """
        Input:
            - nodes: shape = (num_nodes, node_features)
            - start_index, end_index: shape = (num_nodes, )
        """
        # embedding
        start_index, end_index = edge_index
        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(torch.hstack([nodes[start_index],
                                                nodes[end_index]]))

        # message-passing iterations
        for _ in range(self.num_iterations):
            nodes, edges = self._message_step(nodes, edges, start_index, end_index)

        # get prediction input and calculate predition output
        pred_output = {}
        for key, network in self.pred_network.items():
            if key == 'edge':
                pred_input = torch.hstack([nodes[start_index], nodes[end_index], edges])
            elif key == 'node':
                pred_input = nodes
            else:
                raise KeyError(f'Unknown prediction objective {key}!')

            # squeeze the last dimension for scalar output.
            output = network(pred_input)
            if output.shape[-1] == 1:
                output = output.squeeze(1)

            pred_output[key] = output

        return pred_output

    def _message_step(self, nodes, edges, start_index, end_index):
        """
        Message-passing step
        """
        # Compute new edge features
        if self.aggregation == 'add':
            edge_messages = self._get_edge_message_add(edges, end_index, nodes)
        elif self.aggregation == 'max':
            edge_messages = self._get_edge_message_max(edges, end_index, nodes)
        elif self.aggregation == 'add_max':
            edge_messages = self._get_edge_message_add_max(edges, end_index, nodes)
        elif self.aggregation == 'mean_max':
            edge_messages = self._get_edge_message_mean_max(edges, end_index, nodes)
        else:
            raise KeyError(f'Unknown aggregation method {self.aggregation}!')

        node_in = torch.hstack([nodes, edge_messages])
        node_out = self.node_network(node_in) + nodes

        # Compute new edge features
        edge_in = torch.hstack([nodes[start_index], nodes[end_index], edges])
        edge_out = self.edge_network(edge_in) + edges

        return node_out, edge_out

    def _get_edge_message_add(self, edges, end_index, nodes):
        return scatter_add(src      = edges,
                           index    = end_index,
                           dim      = 0,
                           dim_size = nodes.size(0))

    def _get_edge_message_mean(self, edges, end_index, nodes):
        return scatter_mean(src      = edges,
                            index    = end_index,
                            dim      = 0,
                            dim_size = nodes.size(0))

    def _get_edge_message_max(self, edges, end_index, nodes):
        return scatter_max(src      = edges,
                           index    = end_index,
                           dim      = 0,
                           dim_size = nodes.size(0))[0]

    def _get_edge_message_add_max(self, edges, end_index, nodes):
        return torch.hstack([self._get_edge_message_add(edges, end_index, nodes),
                             self._get_edge_message_max(edges, end_index, nodes)])

    def _get_edge_message_mean_max(self, edges, end_index, nodes):
        return torch.hstack([self._get_edge_message_mean(edges, end_index, nodes),
                             self._get_edge_message_max(edges, end_index, nodes)])


def construct_interaction_gnn(model_config):
    """
    Assemble an InteractionGNN
    """
    hidden_features = model_config['hidden']

    # sub-model configurations
    node_encoder_config = model_config['node_encoder']
    edge_encoder_config = model_config['edge_encoder']
    node_network_config = model_config['node_network']
    edge_network_config = model_config['edge_network']

    # configurations control the working of message passing in the GNN
    gnn_config = model_config['gnn']

    # update the configuration parameters that need computation
    edge_encoder_config.update({'input_features': 2 * hidden_features})
    concat_factor = len(gnn_config['aggregation'].split('_')) + 1
    node_network_config.update({'input_features': concat_factor * hidden_features})
    edge_network_config.update({'input_features': 3 * hidden_features})

    pred_network = nn.ModuleDict()
    for key, config in model_config['pred_network'].items():
        if key == 'edge':
            config.update({'input_features': 3 * hidden_features})
        elif key == 'node':
            config.update({'input_features': hidden_features})
        else:
            raise KeyError(f'Unknown prediction objective {key}!')
        pred_network[key] = MLP(**config)

    # construct sub-models
    node_encoder = MLP(**node_encoder_config)
    edge_encoder = MLP(**edge_encoder_config)
    node_network = MLP(**node_network_config)
    edge_network = MLP(**edge_network_config)

    return  InteractionGNN(node_encoder = node_encoder,
                           edge_encoder = edge_encoder,
                           node_network = node_network,
                           edge_network = edge_network,
                           pred_network = pred_network,
                           **gnn_config)


def test():
    with open('configs/config_uniform.yaml', 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    # dummy data
    nodes = torch.randn(1000, 5)
    pairs = torch.randint(0, 1000, size=(2000, 2))
    edge_index = torch.vstack([pairs, pairs.flip(dims=(1,))]).T

    # Assemble GNN
    model_config = config['model']
    gnn = construct_uniform_gnn(model_config)

    with torch.no_grad():
        output = gnn(nodes, edge_index)

    print(output)

if __name__ == '__main__':
    test()
