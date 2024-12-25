import torch
import torchmetrics

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.nn import GATConv, global_mean_pool

import lightning as L


class SpatialPPIv2(L.LightningModule):
    def __init__(self, cfg, return_attention=False) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = cfg['trainer']['lr']
        self.batch_size = cfg['trainer']['batch_size']
        self.return_attention = return_attention
        input_features = cfg['basic']['num_features']

        self.gat1 = GATConv(
            input_features,
            input_features // 2,
            heads=4,
            concat=False
        )
        self.bn1 = nn.BatchNorm1d(input_features // 2)
        self.gat2 = GATConv(
            input_features // 2,
            input_features // 4,
            heads=4,
            concat=False
        )
        self.bn2 = nn.BatchNorm1d(input_features // 4)
        self.gat3 = GATConv(
            input_features // 4,
            input_features // 2,
            heads=4,
            concat=False
        )
        self.bn3 = nn.BatchNorm1d(input_features // 2)
        self.gat4 = GATConv(
            input_features // 2,
            input_features,
            heads=4,
            concat=False
        )
        self.lin = torch.nn.Linear(
            input_features * 4, 1)

    def forward(self, data):
        if self.return_attention:
            node_feature, aw1 = self.gat1(data.x, data.edge_index, data.edge_attr, return_attention_weights=True)
            node_feature = F.relu(self.bn1(node_feature))
            node_feature, aw2 = self.gat2(node_feature, data.edge_index, data.edge_attr, return_attention_weights=True)
            node_feature = F.relu(self.bn2(node_feature))
            node_feature, aw3 = self.gat3(node_feature, data.edge_index, data.edge_attr, return_attention_weights=True)
            node_feature = F.relu(self.bn3(node_feature))
            node_feature, aw4 = self.gat4(node_feature, data.edge_index, data.edge_attr, return_attention_weights=True)
        else:
            node_feature = self.gat1(data.x, data.edge_index, data.edge_attr)
            node_feature = F.relu(self.bn1(node_feature))
            node_feature = self.gat2(node_feature, data.edge_index, data.edge_attr)
            node_feature = F.relu(self.bn2(node_feature))
            node_feature = self.gat3(node_feature, data.edge_index, data.edge_attr)
            node_feature = F.relu(self.bn3(node_feature))
            node_feature = self.gat4(node_feature, data.edge_index, data.edge_attr)

        node_feature = torch.cat([node_feature, data.x], dim=1)
        if data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=int)
            data.data_shape = [data.data_shape]
        
        graph_features = []
        for item_id in torch.unique(data.batch):
            dshape = data.data_shape[item_id]
            residue_feature = node_feature[data.batch == item_id]
            x = [residue_feature[:dshape[0]], residue_feature[dshape[0]:]]
            x = torch.cat([global_mean_pool(i, None) for i in x], dim=1)
            graph_features.append(x)

        output = self.lin(torch.cat(graph_features))
        output = F.sigmoid(output).flatten()

        if self.return_attention:
            return output, [aw1, aw2, aw3, aw4]
        else:
            return output


    def getMatrix(self, preds, labels, name=''):
        values = {
            f"{name}loss": torch.nn.functional.binary_cross_entropy(preds, labels),
            f"{name}acc": torchmetrics.functional.accuracy(preds, labels, 'binary'),
            f"{name}precision": torchmetrics.functional.precision(preds, labels, 'binary'),
            f"{name}recall": torchmetrics.functional.recall(preds, labels, 'binary'),
        }
        return values


    def training_step(self, batch, batch_idx):
        output = self(batch)
        values = self.getMatrix(output, batch.interact.type(torch.float32))
        self.log_dict(values, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True, batch_size=self.batch_size)
        return values['loss']


    def validation_step(self, batch, batch_idx):
        output = self(batch)
        values = self.getMatrix(output, batch.interact.type(torch.float32), 'val_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)


    def test_step(self, batch, batch_idx):
        output = self(batch)
        values = self.getMatrix(output, batch.interact.type(torch.float32), 'test_')
        self.log_dict(values, on_epoch=True, batch_size=self.batch_size)


    def predict_step(self, batch, batch_idx):
        output = self(batch)
        return torch.cat([output.unsqueeze(1), batch.interact.unsqueeze(1)], dim=1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def getModel(cfg, ckpt=None, return_attention=False):
    if ckpt is None:
        return SpatialPPIv2(cfg, return_attention=return_attention)
    else:
        return SpatialPPIv2.load_from_checkpoint(ckpt, cfg=cfg, return_attention=return_attention)
