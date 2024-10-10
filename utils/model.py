import torch
import torchmetrics

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

import lightning as L


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class ResidueEncoder(torch.nn.Module):
    def __init__(self, input_features, hidden_dim):
        super(ResidueEncoder, self).__init__()

        self.conv1 = GATConv(input_features, hidden_dim *
                             2, heads=4, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 8)
        self.conv2 = GCNConv(hidden_dim * 8, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class InterfacePredictor(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.distance_threshold = cfg['dataset']['distance_threshold']
        self.balance_sample = cfg['dataset']['balance_sample']
        self.batch_size = cfg['trainer']['batch_size']
        self.lr = cfg['trainer']['lr']
        self.lr_gamma = cfg['trainer']['lr_gamma']
        self.loss_alpha = cfg['trainer']['loss_alpha']
        self.loss_gamma = cfg['trainer']['loss_gamma']
        self.task = cfg['model']['task']

        self.encoder = ResidueEncoder(
            input_features=cfg['model']['input_features'],
            hidden_dim=cfg['model']['hidden_dim'])
        self.decoder = EdgeDecoder(cfg['model']['hidden_dim'])

    def forward(self, x, edge_index, edge_weight, data_shapes, pred_tars=None):
        node_feature = self.encoder(x, edge_index, edge_weight)

        outputs = []
        start_idx = 0
        for i, data_shape in enumerate(data_shapes):
            if pred_tars is None:
                target = torch.argwhere(torch.ones(data_shape)).T
            else:
                target = torch.tensor(pred_tars[i])

            target += start_idx
            target[1] += data_shape[0]

            result = self.decoder(
                torch.cat([node_feature[j] for j in target], dim=1))
            outputs.append(result.flatten())
            start_idx += sum(data_shape)

        output = torch.flatten(torch.cat(outputs))

        if self.task == 'regression':
            output = F.relu(output)
        elif self.task == 'classification':
            output = F.sigmoid(output)
        return output

    def getMatrix(self, preds, targets, name=''):
        if self.task == 'regression':
            # For regression
            binary_pred = preds < self.distance_threshold
            binary_tar = targets < self.distance_threshold
            # loss = torch.nn.functional.mse_loss(preds, targets)
            loss = torch.nn.functional.huber_loss(preds, targets)
            mae = torchmetrics.functional.mean_absolute_error(preds, targets)

        elif self.task == 'classification':
            # For classification
            binary_tar = (targets < self.distance_threshold).type(
                torch.float32)
            binary_pred = preds
            loss = focal_loss(preds, binary_tar, alpha=self.loss_alpha,
                              gamma=self.loss_gamma, reduction='sum')
            mae = None

        [tn, fp], [fn, tp] = torchmetrics.functional.confusion_matrix(
            binary_pred, binary_tar, 'binary')
        values = {
            f"{name}loss": loss,
            f"{name}acc": torchmetrics.functional.accuracy(binary_pred, binary_tar, 'binary'),
            f"{name}precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            f"{name}recall": tp / (tp + fn) if (tp + fn) > 0 else 0
        }

        if mae is not None:
            values[f"{name}MAE"] = mae

        return values

    def training_step(self, batch, batch_idx):
        # DataBatch(x=[2202, 1024], edge_index=[2, 640322], edge_attr=[640322], y=[321260], data_shape=[4], batch=[2202], ptr=[5])

        pred_tars = batch.target if hasattr(batch, 'target') else None
        pred_y = batch.label if hasattr(batch, 'label') else batch.y

        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            data_shapes=batch.data_shape,
            pred_tars=pred_tars
        )

        values = self.getMatrix(output, pred_y)
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=self.batch_size, sync_dist=True)
        return values['loss']

    def validation_step(self, batch, batch_idx):
        pred_tars = batch.target if hasattr(batch, 'target') else None
        pred_y = batch.label if hasattr(batch, 'label') else batch.y

        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            data_shapes=batch.data_shape,
            pred_tars=pred_tars
        )

        values = self.getMatrix(output, pred_y, name='val_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        pred_tars = batch.target if hasattr(batch, 'target') else None
        pred_y = batch.label if hasattr(batch, 'label') else batch.y

        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            data_shapes=batch.data_shape,
            pred_tars=pred_tars
        )

        values = self.getMatrix(output, pred_y, name='test_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=self.lr_gamma),
            },
        }


class InteractionPredictor(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.gat = GCNConv(
            cfg['model']['input_features'],
            cfg['model']['hidden_dim']
        )
        self.lin = torch.nn.Linear(
            cfg['model']['hidden_dim'] * 2, 1)

        self.lr = cfg['trainer']['lr']
        self.lr_gamma = cfg['trainer']['lr_gamma']
        self.batch_size = cfg['trainer']['batch_size']

    def forward(self, x, edge_index, edge_weight, data_shapes):
        node_feature = self.gat(x, edge_index, edge_weight)

        graph_features = []
        start_idx = 0
        for i, data_shape in enumerate(data_shapes):
            feature_a = global_mean_pool(node_feature[start_idx: start_idx+data_shape[0]], batch=None)
            feature_b = global_mean_pool(
                node_feature[start_idx+data_shape[0]: start_idx+sum(data_shape)], batch=None)
            graph_features.append(self.lin(torch.cat([feature_a, feature_b], dim=1)))
            start_idx += sum(data_shape)

        output = torch.cat(graph_features)
        output = F.sigmoid(output).flatten()
        return output

    def getMatrix(self, preds, targets, name=''):
        [tn, fp], [fn, tp] = torchmetrics.functional.confusion_matrix(
            preds, targets, 'binary')
        values = {
            f"{name}loss": torch.nn.functional.binary_cross_entropy(preds, targets),
            f"{name}acc": torchmetrics.functional.accuracy(preds, targets, 'binary'),
            f"{name}precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            f"{name}recall": tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        return values

    def training_step(self, batch, batch_idx):
        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            data_shapes=batch.data_shape,
        )
        values = self.getMatrix(output, batch.interact.type(torch.float32))
        self.log_dict(values, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True, batch_size=self.batch_size)
        return values['loss']

    def validation_step(self, batch, batch_idx):
        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            data_shapes=batch.data_shape,
        )

        values = self.getMatrix(
            output, batch.interact.type(torch.float32), 'val_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        output = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
        )

        values = self.getMatrix(
            output, batch.interact.type(torch.float32), 'test_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=self.lr_gamma),
            },
        }


class SpatialPPI2(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        ifp_cfg = cfg['InterfacePredictor']
        ppi_cfg = cfg['InteractionPredictor']

        self.interface = InterfacePredictor.load_from_checkpoint(
                ifp_cfg['model']['checkpoint'], cfg=ifp_cfg)
        
        self.interaction = InteractionPredictor.load_from_checkpoint(
                ppi_cfg['model']['checkpoint'], cfg=ppi_cfg)

        self.edge_weight_scale = 1
        self.threshold = ifp_cfg['dataset']['distance_threshold']

        if ifp_cfg['model']['task'] == 'classification':
            self.threshold = ifp_cfg['model']['classification_threshold']
            self.edge_weight_scale = ifp_cfg['dataset']['distance_threshold'] / \
                self.threshold

    def forward(self, x, edge_index, edge_weight, data_shapes):
        matrix = self.interface(x, edge_index, edge_weight, data_shapes)
        idx = 0
        for data_shape in data_shapes:
            pred_matrix = matrix[idx:idx+data_shape[0]*data_shape[1]].reshape(data_shape)
            if self.threshold > 1:
                adj = (self.threshold > pred_matrix) & (pred_matrix > 0)
            else:
                adj = self.threshold < pred_matrix
            pred_edges = torch.nonzero(adj).T
            pred_edges[0] += idx
            pred_edges[0] += idx + data_shape[0]
            edge_index = torch.cat([edge_index, pred_edges], dim=1)
            edge_weight = torch.cat(
                [edge_weight, pred_matrix[adj] * self.edge_weight_scale])
            idx += data_shape[0]*data_shape[1]

        output = self.interaction(x, edge_index, edge_weight, data_shapes)
        return output, matrix
