import dgl
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv, GINConv
from layers import AttentionModule, TensorNetworkModule, MatchingModule, GraphAggregationLayer, OTLayer
from math import exp
from GedMatrix import GedMatrixModule, CostMatrixModule
from ot.gromov import fused_gromov_wasserstein,fused_gromov_wasserstein2

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        # self.bias_model = torch.nn.Linear(2, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def ntn_pass(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        return scores

    def forward(self, data, return_ged=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: pass
        :param predict_value: pass
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        scores = self.ntn_pass(abstract_features_1, abstract_features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = F.relu(self.fully_connected_first(scores))
        scores = F.relu(self.fully_connected_second(scores))
        scores = F.relu(self.fully_connected_third(scores))
        score = torch.sigmoid(self.scoring_layer(scores).view(-1))

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item()

# if training IMDB，using_dropout = false
class GPN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GPN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        if self.args.dataset == 'IMDB':
            using_dropout = False
        else:
            using_dropout = self.training
        # using_dropout = False
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        tmp_feature_1 = abstract_features_1
        tmp_feature_2 = abstract_features_2

        abstract_features_1 = torch.sub(tmp_feature_1, self.matching_2(tmp_feature_2))
        abstract_features_2 = torch.sub(tmp_feature_2, self.matching_1(tmp_feature_1))

        abstract_features_1 = torch.abs(abstract_features_1)
        abstract_features_2 = torch.abs(abstract_features_2)

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item()

# if training IMDB，using_dropout = false
class GedGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GedGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        self.costMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        # self.bias_model = torch.nn.Linear(2, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        #using_dropout = self.training
        #using_dropout = False
        if self.args.dataset == 'IMDB':
            using_dropout = False
        else:
            using_dropout = self.training
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=using_dropout)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=using_dropout)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)

        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        soft_matrix = m(map_matrix) * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        #print(soft_matrix.sum(), torch.sigmoid(soft_matrix.sum()))
        #print(bias_value, torch.sigmoid(bias_value))
        #print(soft_matrix.sum() + bias_value, torch.sigmoid(soft_matrix.sum() + bias_value))
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix


class TaGSim(torch.nn.Module):
    """
    TaGSim: Type-aware Graph Similarity Learning and Computation
    https://github.com/jiyangbai/TaGSim
    """
    def __init__(self, args, number_of_labels):
        super(TaGSim, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        self.tensor_network_nc = TensorNetworkModule(self.args, 2 * self.number_labels)
        self.tensor_network_in = TensorNetworkModule(self.args, 2 * self.number_labels)
        self.tensor_network_ie = TensorNetworkModule(self.args, 2 * self.number_labels)

        self.fully_connected_first_nc = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_nc = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_nc = torch.nn.Linear(8, 4)
        self.scoring_layer_nc = torch.nn.Linear(4, 1)

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_in = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_in = torch.nn.Linear(8, 4)
        self.scoring_layer_in = torch.nn.Linear(4, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ie = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ie = torch.nn.Linear(8, 4)
        self.scoring_layer_ie = torch.nn.Linear(4, 1)

    def gal_pass(self, edge_index, features):
        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        n1, n2 = data["n1"], data["n2"]

        adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.shape[1]), (n1, n1)).to_dense()
        adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.shape[1]), (n2, n2)).to_dense()
        # remove self-loops
        adj_1 = adj_1 * (1.0 - torch.eye(n1))
        adj_2 = adj_2 * (1.0 - torch.eye(n2))

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)

        graph1_01concat = torch.cat([features_1, graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([graph1_hidden1, graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([graph2_hidden1, graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)

        scores_nc = self.tensor_network_nc(graph1_01pooled, graph2_01pooled)
        scores_nc = torch.t(scores_nc)

        scores_nc = torch.nn.functional.relu(self.fully_connected_first_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_second_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_third_nc(scores_nc))
        score_nc = torch.sigmoid(self.scoring_layer_nc(scores_nc))

        scores_in = self.tensor_network_in(graph1_01pooled, graph2_01pooled)
        scores_in = torch.t(scores_in)

        scores_in = torch.nn.functional.relu(self.fully_connected_first_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_second_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_third_in(scores_in))
        score_in = torch.sigmoid(self.scoring_layer_in(scores_in))

        scores_ie = self.tensor_network_ie(graph1_12pooled, graph2_12pooled)
        scores_ie = torch.t(scores_ie)

        scores_ie = torch.nn.functional.relu(self.fully_connected_first_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_second_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_third_ie(scores_ie))
        score_ie = torch.sigmoid(self.scoring_layer_ie(scores_ie))

        score = torch.cat([score_nc.view(-1), score_in.view(-1), score_ie.view(-1)])
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.sum().item()


class GEDIOT(torch.nn.Module):
    def __init__(self,args,number_of_labels):
        super(GEDIOT,self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def init_mlp_features(self):
        k = self.number_labels+self.args.filters_1+self.args.filters_2+self.args.filters_3
        layers = []
        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k,self.args.filters_3))
        self.mlp = torch.nn.Sequential(*layers)
    
    def setup_layers(self):
        self.args.gnn_operator = 'gin'
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin': 
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        self.init_mlp_features()
       
        self.costMatrix = CostMatrixModule(self.args.filters_3)
        self.mapMatrix = OTLayer()
        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)    
          
    def convolutional_pass(self, edge_index, features):
        features_list = features
        features = self.convolution_1(features, edge_index)#GIN1
        features_list = torch.cat([features_list,features],dim=1)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features_list = torch.cat([features_list,features],dim=1)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        features_list = torch.cat([features_list,features],dim=1)
        features = self.mlp(features_list)
        # features = torch.sigmoid(features)
        return features
        
    def unmatching_part(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1) #d x 1
        pooled_features_2 = self.attention(abstract_features_2) #d x 1
        scores = self.tensor_network(pooled_features_1, pooled_features_2) # tn x 1
        scores = torch.t(scores) # 1 x tn
       
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score   
    
    def forward(self,data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)

        cost_matrix = torch.nn.functional.tanh(cost_matrix)
        
        map_matrix=self.mapMatrix(cost_matrix)
       
        soft_matrix = map_matrix * cost_matrix #按元素乘法
        bias_value = self.unmatching_part(abstract_features_1, abstract_features_2)

        score = torch.sigmoid(soft_matrix.sum() + bias_value)              
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix  

class GEDGW(object):
    def __init__(self,data,args):
        super(GEDGW, self).__init__()
        self.n1=data["n1"]
        self.n2=data["n2"]
        self.g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=self.n1)
        self.g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=self.n2)
        self.g1.ndata['f'] = data["features_1"]
        self.g2.ndata['f'] = data["features_2"]
        self.args=args
        self.set_cost_test()
    def set_cost(self):
        self.nu = torch.ones(self.n2)
        self.mu = torch.ones(self.n1)
        assert self.g1.ndata['f'].shape[1]==self.g2.ndata['f'].shape[1]
        if self.args.dataset == 'AIDS':
            assert self.g1.ndata['f'].shape[1] == 29
        feature1 = self.g1.ndata['f'].unsqueeze(1)
        feature2 = self.g2.ndata['f'].unsqueeze(0)
        self.cross_cost = torch.eq(feature1, feature2).all(dim=-1).float()
        self.cost1 = self.g1.adj().to_dense().float()-torch.eye(self.n1)
        self.cost2 = self.g2.adj().to_dense().float()-torch.eye(self.n2)
        if self.n1 < self.n2:
            self.mu = torch.cat((self.mu, torch.tensor([self.n2-self.n1])), dim=0)
            padding = (0, 1, 0, 1)
            self.cost1 = F.pad(self.cost1, padding, "constant", 0)
            padding1 = (0, 0, 0, 1)
            self.cross_cost = F.pad(self.cross_cost, padding1, "constant", 0)
        self.nu = self.nu/self.n2
        self.mu = self.mu/self.n2
        self.cost1=self.cost1*self.n2
        self.cost2=self.cost2*self.n2
        self.cross_cost = (1-self.cross_cost)*self.n2
    
    #square test
    def set_cost_test(self):
        self.nu = torch.ones(self.n2)
        self.mu = torch.ones(self.n2)
        assert self.g1.ndata['f'].shape[1]==self.g2.ndata['f'].shape[1]
        if self.args.dataset == 'AIDS':
            assert self.g1.ndata['f'].shape[1] == 29
        feature1 = self.g1.ndata['f'].unsqueeze(1)
        feature2 = self.g2.ndata['f'].unsqueeze(0)
        self.cross_cost = torch.eq(feature1, feature2).all(dim=-1).float()
        self.cost1 = self.g1.adj().to_dense().float()-torch.eye(self.n1)
        self.cost2 = self.g2.adj().to_dense().float()-torch.eye(self.n2)
        if self.n1 < self.n2:
            delta = self.n2-self.n1
            #self.mu = torch.cat((self.mu, torch.tensor([self.n2-self.n1])), dim=0)
            padding = (0, delta, 0, delta)
            self.cost1 = F.pad(self.cost1, padding, "constant", 0)
            padding1 = (0, 0, 0, delta)
            self.cross_cost = F.pad(self.cross_cost, padding1, "constant", 0)
        assert self.nu.shape == self.mu.shape
        assert self.cross_cost.shape[0]==self.cross_cost.shape[1]
        assert self.cost1.shape==self.cost2.shape
        self.nu = self.nu/self.n2
        self.mu = self.mu/self.n2
        self.cost1=self.cost1*self.n2
        self.cost2=self.cost2*self.n2
        self.cross_cost = (1-self.cross_cost)*self.n2
    
    def process(self):
        alpha_test = 1/3
        reverse = 3/2
        T_cg, log_cg = fused_gromov_wasserstein(
            self.cross_cost, self.cost1,self.cost2, self.mu, self.nu, 'square_loss',alpha=alpha_test,armijo=True,verbose=False, log=True)
        print(f"DEBUG: fgw_dist brute = {log_cg['fgw_dist']}") # Ligne de débogage
        pre_ged = log_cg['fgw_dist']*reverse
        return T_cg[:self.n1,:], pre_ged.item()
