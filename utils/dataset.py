import os
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch_geometric.data import Dataset

from .tool import Embed, pdb2data, extractPDB


class ProteinIntDataset(Dataset):
    def __init__(
            self, 
            split, 
            cfg,
            usage=None,
            balance_sample=False,
    ):
        self.embeder = Embed(cfg['embedding'])
        self.distance_threshold = cfg['distance_threshold']
        self.data_root = cfg['data_root']

        self.pairs = pd.read_csv(cfg[split], index_col=0).to_dict('records')

        self.usage = usage
        if balance_sample:
            # Sampling
            self.balance_sample = balance_sample
            self.negative_by_positive = cfg['negative_by_positive']
            self.nsample = int(cfg['nsample'] / 2)
            super().__init__(transform=self.load_data)
        else:
            super().__init__()

    def get(self, idx):
        rec = self.pairs[idx]

        if rec.get('pdb_gt', '-') != '-':
            chains = [rec['chain_a'], rec['chain_b']]
            pdb_path = os.path.join(self.data_root, rec['pdb_gt'])
            seqs, coords = extractPDB(pdb_path, chains)

        elif rec['pdb_a'] == rec['pdb_b']:
            chains = [rec['chain_a'], rec['chain_b']]
            pdb_path = os.path.join(self.data_root, rec['pdb_a'])
            seqs, coords = extractPDB(pdb_path, chains)

        else:
            pdb_path_a = os.path.join(self.data_root, rec['pdb_a'])
            seq_a, coord_a = extractPDB(pdb_path_a, rec['chain_a'])
            pdb_path_b = os.path.join(self.data_root, rec['pdb_b'])
            seq_b, coord_b = extractPDB(pdb_path_b, rec['chain_b'])
            seqs = [seq_a, seq_b]
            coords = [coord_a, coord_b]

        data = pdb2data(
            seqs,
            coords,
            distance_threshold=self.distance_threshold,
            embeder=self.embeder,
            interact=rec['label'],
            usage=self.usage
        )
        return data

    def len(self):
        return len(self.pairs)

    def load_data(self, data):
        if self.balance_sample:
            y = data.y.reshape(data.data_shape)

            pos = torch.argwhere(y <= self.distance_threshold)
            neg = torch.argwhere(y > self.distance_threshold)
            nsample = min([self.nsample, pos.size(0), neg.size(0)])
            nneg = min(self.negative_by_positive * nsample, neg.size(0))

            pos = pos[torch.randperm(pos.size(0))[:nsample]]
            neg = neg[torch.randperm(neg.size(0))[:nneg]]
            targets = torch.cat([pos, neg]).T

            data.target = targets.tolist()
            data.label = y[targets[0], targets[1]]

        return data
