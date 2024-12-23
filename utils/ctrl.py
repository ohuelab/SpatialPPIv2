import os
import random
import torch
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.data import OnDiskDataset
import sklearn.metrics as metrics

from .model import getModel
from .dataset import getDataset


def train(cfg, ckpt=False):
    model = getModel(cfg, ckpt=cfg['model']['checkpoint'] if ckpt else None)
    train = getDataset(cfg, 'train')
    val = getDataset(cfg, 'val')
    test = getDataset(cfg, 'test')

    train_loader = DataLoader(
        train,
        batch_size=cfg['trainer']['batch_size'],
        shuffle=True,
        num_workers=cfg['trainer']["loader_workers"]
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg['trainer']['batch_size'],
        shuffle=False,
        num_workers=cfg['trainer']["test_workers"]
    )
    print(
        f'Dataset loaded, train {len(train)}, val {len(val)}, test {len(test)}')
    
    trainer = L.Trainer(
        accelerator=cfg['trainer']["accelerator"],
        devices=cfg['trainer']["devices"],
        max_epochs=cfg['trainer']["max_epochs"],
        check_val_every_n_epoch=cfg['trainer']["check_val_every_n_epoch"],
        accumulate_grad_batches=cfg['trainer']["accumulate_grad_batches"],
    )

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    test_loader = DataLoader(
        test,
        batch_size=cfg['trainer']['batch_size'],
        shuffle=False,
        num_workers=cfg['trainer']["test_workers"]
    )
    # Test model
    model.eval()
    trainer.test(model, dataloaders=test_loader)


def eval(cfg, ckpt=False):
    model = getModel(cfg, ckpt=ckpt)
    test = getDataset(cfg, 'test')
    model.eval()
    print(f'Dataset loaded, test {len(test)}')

    test_loader = DataLoader(
        test,
        batch_size=cfg['trainer']['batch_size'],
        shuffle=False,
        num_workers=cfg['trainer']["loader_workers"]
    )
    trainer = L.Trainer(
        accelerator=cfg['trainer']["accelerator"],
        devices=cfg['trainer']["devices"],
    )

    # Test model
    output = trainer.predict(model, test_loader)
    pred, label = torch.cat(output).T

    evaluation_matrix = {
        'Accuracy': metrics.accuracy_score(label, pred > 0.5),
        'Precision': metrics.precision_score(label, pred > 0.5),
        'Recall': metrics.recall_score(label, pred > 0.5),
        'F1 Score': metrics.f1_score(label, pred > 0.5),
        'Average Precision': metrics.average_precision_score(label, pred),
        'AUC-ROC': metrics.roc_auc_score(label, pred)
    }
    print(pd.Series(evaluation_matrix).round(3))
    return pred, label

