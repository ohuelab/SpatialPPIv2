import random
import torch
import lightning as L
from torch_geometric.loader import DataLoader

from .tool import getConfig
from .model import InterfacePredictor, InteractionPredictor
from .dataset import ProteinIntDataset


def train_interface(cfg_path, resume=False):
    cfg = getConfig(cfg_path)['InterfacePredictor']

    data_cfg = cfg['dataset']
    model_cfg = cfg['model']
    trainer_cfg = cfg['trainer']

    if resume:
        model = InterfacePredictor.load_from_checkpoint(
            model_cfg['checkpoint'], cfg=cfg)
    else:
        model = InterfacePredictor(cfg=cfg)
    model.train()

    train = ProteinIntDataset(
        "train",
        data_cfg,
        balance_sample=data_cfg['balance_sample']
    )
    val = ProteinIntDataset(
        'val',
        data_cfg
    )
    test = ProteinIntDataset(
        'test',
        data_cfg)
    print(f'Dataset loaded, train {len(train)}, val {len(val)}')

    train_loader = DataLoader(
        train,
        batch_size=trainer_cfg["batch_size"],
        shuffle=True,
        num_workers=trainer_cfg["loader_workers"]
    )
    val_loader = DataLoader(
        val,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False,
        num_workers=trainer_cfg["test_workers"]
    )
    test_loader = DataLoader(
        test,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False,
        num_workers=trainer_cfg["test_workers"]
    )

    trainer = L.Trainer(
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        max_epochs=trainer_cfg["max_epochs"],
        check_val_every_n_epoch=trainer_cfg["check_val_every_n_epoch"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
    )

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Test model
    model.eval()
    trainer.test(model, dataloaders=test_loader)


def eval_interface(cfg_path):
    cfg = getConfig(cfg_path)['InterfacePredictor']

    data_cfg = cfg['dataset']
    model_cfg = cfg['model']
    trainer_cfg = cfg['trainer']

    model = InterfacePredictor.load_from_checkpoint(
        model_cfg['checkpoint'], cfg=cfg)
    model.eval()
    test = ProteinIntDataset('test', data_cfg)
    print(f'Dataset loaded, test {len(test)}')

    test_loader = DataLoader(
        test,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False,
        num_workers=trainer_cfg["loader_workers"]
    )
    trainer = L.Trainer(
        default_root_dir=model_cfg['checkpoint'][:model_cfg['checkpoint'].find(
            'checkpoints')],
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"]
    )

    # Test model
    trainer.test(model, dataloaders=test_loader)


@torch.no_grad()
def inference_interface(cfg_path, target=None):
    cfg = getConfig(cfg_path)['InterfacePredictor']

    data_cfg = cfg['dataset']
    model_cfg = cfg['model']

    model = InterfacePredictor.load_from_checkpoint(
        model_cfg['checkpoint'], cfg=cfg)
    model.eval()

    if target is None:
        print("Randomly inferencing for test")
        val = ProteinIntDataset('val', data_cfg)
        target_index = random.randint(0, len(val))
        print(f"The {target_index} in val set was selected.")
        target = val[target_index]

    print('----------Inference----------')
    print('Target data:', target)
    target.to(model.device)

    output = model(
        x=target.x,
        edge_index=target.edge_index,
        edge_weight=target.edge_attr,
        data_shapes=[target.data_shape],
        pred_tars=[target.target] if hasattr(target, 'target') else None
    )

    label = target.label if hasattr(target, 'label') else target.y

    matrix = model.getMatrix(output, label)

    print('Output:', output.round(decimals=2))
    print('Label:', label)

    for i in matrix:
        print(i, matrix[i])


def example_interface(cfg_path):
    cfg = getConfig(cfg_path)
    data_cfg = cfg['InterfacePredictor']['dataset']
    data = ProteinIntDataset('train', data_cfg)[0]
    inference_interface(cfg_path, data)


def train_interaction(cfg_path, resume=False):
    cfg = getConfig(cfg_path)['InteractionPredictor']

    if resume:
        model = InteractionPredictor.load_from_checkpoint(
            cfg['model']['checkpoint'],
            cfg=cfg
        )
    else:
        model = InteractionPredictor(cfg)

    train = ProteinIntDataset("train", cfg['dataset'], usage='interaction')
    val = ProteinIntDataset("val", cfg['dataset'], usage='interaction')
    test = ProteinIntDataset("test", cfg['dataset'], usage='interaction')

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
        f'Dataset loaded, train {len(train)}, val {len(val)}, test{len(test)}')

    trainer = L.Trainer(
        accelerator=cfg['trainer']["accelerator"],
        devices=cfg['trainer']["devices"],
        max_epochs=cfg['trainer']["max_epochs"],
        check_val_every_n_epoch=cfg['trainer']["check_val_every_n_epoch"],
        accumulate_grad_batches=cfg['trainer']["accumulate_grad_batches"],
        # callbacks=[ModelCheckpoint(verbose=True, every_n_train_steps=500)]
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


def eval_interaction(cfg_path):
    cfg = getConfig(cfg_path)['InteractionPredictor']
    model = InteractionPredictor.load_from_checkpoint(
        cfg['model']['checkpoint'],
        cfg=cfg
    )
    model.eval()
    test = ProteinIntDataset("test", cfg['dataset'], usage='interaction')
    log_dir = cfg['model']['checkpoint'][:cfg['model']['checkpoint'].find('checkpoints')]
    print(f'Dataset loaded, test {len(test)}')

    test_loader = DataLoader(
        test,
        batch_size=cfg['trainer']["batch_size"],
        shuffle=False,
        num_workers=cfg['trainer']["loader_workers"]
    )
    trainer = L.Trainer(
        default_root_dir=log_dir,
        accelerator=cfg['trainer']["accelerator"],
        devices=cfg['trainer']["devices"]
    )
    trainer.test(model, dataloaders=test_loader)
