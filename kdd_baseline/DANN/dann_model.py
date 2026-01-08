"""
Complete implementation of Domain-Adversarial Training of Neural Networks (DANN)
Based on: Ganin et al., 2015 - "Domain-Adversarial Training of Neural Networks"
https://www.jmlr.org/papers/volume17/15-239/15-239.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) from DANN paper.
    Forward pass: identity transformation
    Backward pass: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer wrapper"""
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = 1.0

    def set_lambda(self, lambda_):
        """Set the lambda parameter for gradient reversal"""
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GNNFeatureExtractor(nn.Module):
    """
    Graph Neural Network Feature Extractor using GIN (Graph Isomorphism Network)
    or GCN (Graph Convolutional Network)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, gnn_type='gin', dropout=0.5):
        super(GNNFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if gnn_type == 'gin':
            nn_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer))
        else:  # gcn
            self.convs.append(GCNConv(input_dim, hidden_dim))

        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == 'gin':
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(nn_layer))
            else:  # gcn
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_add_pool(x, batch)
        return x


class LabelClassifier(nn.Module):
    """Label Classifier (predicts class labels)"""
    def __init__(self, feature_dim, num_classes, hidden_dim=128, dropout=0.5):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class DomainClassifier(nn.Module):
    """Domain Classifier (predicts domain labels)"""
    def __init__(self, feature_dim, num_domains, hidden_dim=128, dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_domains)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)
        return x


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN)

    Architecture:
        - Feature Extractor (GNN): Learns domain-invariant features
        - Label Classifier: Predicts class labels
        - Domain Classifier: Predicts domain labels (with gradient reversal)

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        num_domains: Number of domains
        hidden_dim: Hidden dimension for feature extractor
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gin' or 'gcn')
        dropout: Dropout rate
    """
    def __init__(self, input_dim, num_classes, num_domains,
                 hidden_dim=128, num_layers=3, gnn_type='gin', dropout=0.5):
        super(DANN, self).__init__()

        # Feature extractor
        self.feature_extractor = GNNFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        # Label classifier
        self.label_classifier = LabelClassifier(
            feature_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer()

        # Domain classifier
        self.domain_classifier = DomainClassifier(
            feature_dim=hidden_dim,
            num_domains=num_domains,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x, edge_index, batch, alpha=1.0):
        """
        Forward pass

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            alpha: Lambda parameter for gradient reversal (default: 1.0)

        Returns:
            class_output: Class predictions
            domain_output: Domain predictions
            features: Extracted features
        """
        # Extract features
        features = self.feature_extractor(x, edge_index, batch)

        # Class prediction
        class_output = self.label_classifier(features)

        # Domain prediction (with gradient reversal)
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)

        return class_output, domain_output, features

    def predict(self, x, edge_index, batch):
        """Predict class labels without domain classification"""
        features = self.feature_extractor(x, edge_index, batch)
        class_output = self.label_classifier(features)
        return class_output


def get_lambda_alpha(epoch, max_epochs):
    """
    Calculate lambda parameter for gradient reversal as in DANN paper.
    Lambda increases from 0 to 1 following the schedule:
    lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    where p progresses linearly from 0 to 1, gamma = 10
    """
    p = float(epoch) / float(max_epochs)
    gamma = 10.0
    lambda_p = 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0
    return lambda_p.item()
