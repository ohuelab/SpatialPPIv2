import os
import json
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset, OnDiskDataset
from torch_geometric.loader import DataLoader
from scipy.spatial import distance_matrix

from .tool import Embed, extractPDB


class ProteinIntDataset(Dataset):
    def __init__(
            self,
            path,
            data_root,
            embedding,
            distance_threshold,
            transform=None,
            use_cuda=False
    ):
        self.embeder = Embed(embedding, use_cuda if use_cuda else 'cpu')
        self.distance_threshold = distance_threshold
        self.data_root = data_root

        self.pairs = pd.read_csv(path, index_col=0).to_dict('records')
        super().__init__(transform=transform)

    def get(self, idx):
        rec = self.pairs[idx]

        pdb_path_a = os.path.join(self.data_root, rec['pdb_a'])
        seq_a, coord_a = extractPDB(pdb_path_a, rec['chain_a'])
        pdb_path_b = os.path.join(self.data_root, rec['pdb_b'])
        seq_b, coord_b = extractPDB(pdb_path_b, rec['chain_b'])
        seqs = [seq_a, seq_b]
        coords = [coord_a, coord_b]
        
        data = Data(
            x=torch.cat([self.embeder.encode(s) for s in seqs]),
            coords=coords,
            interact=rec['label']
        )
        return data
        
    def len(self):
        return len(self.pairs)


class SequenceDataset(Dataset):
    def __init__(
            self,
            path,
            embedding,
            distance_threshold,
            use_cuda=False
    ):
        self.embeder = Embed(embedding, use_cuda if use_cuda else 'cpu')
        self.distance_threshold = distance_threshold

        self.pairs = pd.read_csv(path, index_col=0).to_dict('records')
        super().__init__()

    def get(self, idx):
        rec = self.pairs[idx]

        seq_a, adj_a = self.embeder.encode(rec['seq_a'], attention_contact=True)
        seq_b, adj_b = self.embeder.encode(rec['seq_b'], attention_contact=True)
        data = build_data_from_adj(
            features=[seq_a, seq_b],
            adjs=[adj_a, adj_b],
            interact=rec['label'], 
            distance_threshold=self.distance_threshold
        )
        return data
        
    def len(self):
        return len(self.pairs)


def saveDatasetOnDisk(dataset, saveroot, workers=1):
    exist = False
    if os.path.exists(saveroot):
        print('File exists!')
        exist = True

    saving = OnDiskDataset(root=saveroot)
    
    start_id = 0 if not exist else len(saving)
    dataset = dataset[start_id:]

    if workers==1:
        for data in tqdm(dataset):
            rec = Data(
                x = data.x,
                coords=data.coords[0],
                interact=data.interact[0]
            )
            saving.append(data)
    
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers
        )

        for data in tqdm(dataloader):
            rec = Data(
                x = data.x,
                coords=data.coords[0],
                interact=data.interact[0]
            )
            saving.append(rec)


def build_data(node_feature, coords, interact=0, distance_threshold=8):
    coord_a, coord_b = coords
    coord_ab = np.concatenate(coords)
    distanceMatrix = torch.Tensor(distance_matrix(coord_ab, coord_ab))
    dshape = [len(coord_a), len(coord_b)]
    # contact_gt = distanceMatrix[:dshape[0], dshape[0]:].clone()

    distanceMatrix[:dshape[0], dshape[0]:] = distance_threshold
    distanceMatrix[dshape[0]:, :dshape[0]] = distance_threshold
    
    adj = (distanceMatrix <= distance_threshold) & (distanceMatrix > 0)
    edge_index = torch.nonzero(adj).T
    edge_attr = distanceMatrix[adj].type(torch.float32)

    data = Data(
        x=node_feature,
        data_shape=dshape,
        edge_index=edge_index,
        edge_attr=edge_attr,
        interact=interact,
        # contact_gt=contact_gt
    )
    return data


def build_data_from_adj(features, adjs, interact=0, distance_threshold=8):
    dshape = [features[0].shape[0], features[1].shape[0]]
    distanceMatrix = torch.zeros([sum(dshape), sum(dshape)], dtype=torch.float32)
    distanceMatrix[:dshape[0], :dshape[0]] = (adjs[0] > 0.5) * distance_threshold
    distanceMatrix[dshape[0]:, dshape[0]:] = (adjs[1] > 0.5) * distance_threshold
    distanceMatrix[:dshape[0], dshape[0]:] = distance_threshold
    distanceMatrix[dshape[0]:, :dshape[0]] = distance_threshold

    adj = distanceMatrix > 0
    edge_index = torch.nonzero(adj).T
    edge_attr = distanceMatrix[adj].type(torch.float32)

    data = Data(
        x=torch.cat(features),
        data_shape=dshape,
        edge_index=edge_index,
        edge_attr=edge_attr,
        interact=interact,
    )
    return data


def getDataset(cfg, split):
    dataset_type = cfg['dataset'][split]['type']

    def transformer(data):
        grpData = build_data(
            node_feature=data.x,
            coords=data.coords,
            interact=data.interact,
            distance_threshold=cfg['basic']['distance_threshold']
        )
        return grpData

    if dataset_type == 'ondisk':
        dataset = OnDiskDataset(
            root=cfg['dataset'][split]['path'], 
            transform=transformer
        )

    elif dataset_type == 'csv':
        dataset = ProteinIntDataset(
            path=cfg['dataset'][split]['path'], 
            data_root=cfg['dataset'][split]['data_root'],
            embedding=cfg['basic']['embedding'],
            distance_threshold=cfg['basic']['distance_threshold'],
            transform=transformer,
            use_cuda=cfg['basic']['embedder_device']
        )

    elif dataset_type == 'csv_esmac':
        dataset = SequenceDataset(
            path=cfg['dataset'][split]['path'], 
            embedding=cfg['basic']['embedding'],
            distance_threshold=cfg['basic']['distance_threshold'],
            use_cuda=cfg['basic']['embedder_device']
        )

    return dataset
