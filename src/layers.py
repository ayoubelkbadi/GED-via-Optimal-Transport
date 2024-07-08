"""Classes for SimGNN modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args, input_dim=None):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = self.args.filters_3 if (input_dim is None) else input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim,
                                                             self.input_dim,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(self.input_dim, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class Mlp(torch.nn.Module):
    def __init__(self, dim):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(Mlp, self).__init__()

        self.dim = dim
        layers = []
        '''
        while dim > 1:
            layers.append(torch.nn.Linear(dim, dim // 2))
            layers.append(torch.nn.ReLU())
            dim = dim // 2
        layers[-1] = torch.nn.Sigmoid()
        '''

        layers.append(torch.nn.Linear(dim, dim * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dim * 2, dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dim, 1))
        #layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


# from noah
class MatchingModule(torch.nn.Module):
    """
    Graph-to-graph Module to gather cross-graph information.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MatchingModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN/GIN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.sum(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        return transformed_global


#from TaGSim
class GraphAggregationLayer(nn.Module):

    def __init__(self, in_features=10, out_features=10):
        super(GraphAggregationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input, adj):
        h_prime = torch.mm(adj, input)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.shape)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.shape
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros(shape).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

class OTLayer(nn.Module):
    def __init__(self, max_iter: int=5):
        super(OTLayer, self).__init__()
        self.max_iter = max_iter
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        #self.epsilon = torch.zeros(1)
        self.sinkhorn = PSinkhorn()
    
    def forward(self, cost_matrix):
        K = -cost_matrix/(self.epsilon+0.05)
        match = self.sinkhorn(K,self.max_iter)
        return match


class PSinkhorn(nn.Module):
    def __init__(self):
        super().__init__()    
    def forward(self, corr,max_iter,eps=1e-16):
        n1,n2 = corr.shape
        assert n1<=n2
        dummy_mar = torch.ones(1,requires_grad=False)
        dummy_flag = False
        if n1<n2:
            dummy_shape = list(corr.shape)
            dummy_shape[0]=1
            corr = torch.cat((corr,torch.full(dummy_shape, 0.0)),dim=0)
            dummy_mar = dummy_mar+(n2-n1-1)
            dummy_flag = True
        n11,n22 = corr.shape
        if dummy_flag:
            assert n11==(n1+1)
        log_prob1 = (
            torch.zeros(n11)
        )
        log_prob2 = (
            torch.zeros(n22)
        )
        log_prob1[-1]=log_prob1[-1]+torch.log(dummy_mar)
        log_u = torch.zeros_like(log_prob1)
        # ot
        for _ in range(max_iter):
            
            log_v = log_prob2-torch.logsumexp(corr + log_u.unsqueeze(-1), -2)
            log_u = log_prob1-torch.logsumexp(corr + log_v.unsqueeze(-2), -1)
            
        T = torch.exp(log_u.unsqueeze(-1) + corr + log_v.unsqueeze(-2))
        if dummy_flag:
            T = T[:-1,:]
        assert n1 ==T.shape[0] and n2 ==T.shape[1]
        return T