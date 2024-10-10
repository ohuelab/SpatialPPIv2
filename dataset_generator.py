import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance_matrix

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from pinder.core import get_index

from utils.tool import extractPDB, getConfig


class DataChecker(Dataset):
    def __init__(
            self,
            pdblists,
            cfg
    ):
        self.distance_threshold = cfg['distance_threshold']
        self.data_root = cfg['data_root']
        self.pdblists = pdblists

        super().__init__()

    def get(self, idx):
        pdb_path = os.path.join(self.data_root, self.pdblists[idx])
        if not os.path.exists(pdb_path):
            return Data(x=[-1, -1, -1])
        seqs, coords = extractPDB(pdb_path, ['L', 'R'])
        if len(seqs) != 2 or len(coords) != 2:
            return Data(x=[-1, -1, -1])
        seq_a, seq_b = seqs
        coord_a, coord_b = coords
        if len(seq_a) != len(coord_a) or len(seq_b) != len(coord_b):
            return Data(x=[-1, -1, -1])
        contacts = (distance_matrix(coord_a, coord_b)
                    < self.distance_threshold).sum()
        return Data(x=[len(seq_a), len(seq_b), int(contacts)])

    def len(self):
        return len(self.pdblists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/tsubame.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--biogrid", default="")
    args = parser.parse_args()

    # Load configs
    cfg = getConfig(args.config)['InterfacePredictor']['dataset']

    index = get_index().copy()
    index = index[index['split'] == args.split]
    pdblist = index['id'].to_numpy() + '.pdb'
    dataset = DataChecker(pdblist, cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # filter pinder dataset
    print('Filtering pinder dataset')
    results = []
    for data in tqdm(loader):
        results.append(data.x[0])
    results = np.array(results).T

    positive = pd.DataFrame({
        'jname': index['uniprot_L'].to_numpy() + '-' + index['uniprot_R'].to_numpy(),
        'label': np.ones(len(pdblist), dtype=int),
        'uniprot_a': index['uniprot_L'].to_numpy(),
        'uniprot_b': index['uniprot_R'].to_numpy(),
        'seq_a_len': results[0],
        'seq_b_len': results[1],
        'ncontacts': results[2],
        'pdb_a': pdblist,
        'pdb_b': pdblist,
        'pdb_gt': pdblist,
        'chain_a': ['L'] * len(pdblist),
        'chain_b': ['R'] * len(pdblist)
    })

    positive.to_csv(f'datasets/origin_pinder_{args.split}.csv')

    positive = positive[(positive['seq_a_len'] > 35) &
                        (positive['seq_a_len'] < 300)]
    positive = positive[(positive['seq_b_len'] > 35) &
                        (positive['seq_b_len'] < 300)]
    positive = positive[positive['ncontacts'] > 8]

    positive.to_csv(f'datasets/filtered_pinder_{args.split}.csv')
    print('Positive dataset saved')

    print('Get all proteins from positive dataset')
    uniprot_a = positive.drop_duplicates('uniprot_a', keep='first').get(
        ['uniprot_a', 'pdb_gt', 'chain_a', 'seq_a_len']).copy().rename(
            columns={'uniprot_a': 'uniprot', 'chain_a': 'chain', 'seq_a_len': 'seqlen'})
    uniprot_b = positive.drop_duplicates('uniprot_b', keep='first').get(
        ['uniprot_b', 'pdb_gt', 'chain_b', 'seq_b_len']).copy().rename(
            columns={'uniprot_b': 'uniprot', 'chain_b': 'chain', 'seq_a_len': 'seqlen'})
    uniprot = pd.concat([uniprot_a, uniprot_b]).drop_duplicates(
        'uniprot', keep='first')
    prot2st = pd.DataFrame(uniprot.get(
        ['uniprot', 'pdb_gt', 'chain' ,'seqlen']).to_numpy(), index=uniprot['uniprot'].to_numpy())
    prot2st = prot2st[~prot2st[1].str.contains(
        'UNDEFINED')]

    print('Load BioGRID database')
    biogrid_ds = pd.read_csv(args.biogrid, delimiter='\t')
    protkey_a = 'SWISS-PROT Accessions Interactor A'
    protkey_b = 'SWISS-PROT Accessions Interactor B'
    concater = '^'
    valid = (biogrid_ds[protkey_a] != '-') & (biogrid_ds[protkey_b] != '-')
    biogrid_ds = biogrid_ds[valid]
    valid = biogrid_ds[protkey_a].isin(
        prot2st.index) & biogrid_ds[protkey_b].isin(prot2st.index)
    biogrid_ds = biogrid_ds[valid]
    biogrid = biogrid_ds.get([protkey_a, protkey_b]).to_numpy()
    biogrid = biogrid.T
    biogrid = np.unique(np.concatenate([
        biogrid[0] + concater + biogrid[1],
        biogrid[1] + concater + biogrid[0]
    ]))
    biogrid = pd.Series(biogrid)

    # Generate Negative Samples
    print('Generating negative samples')
    np.random.seed(2032)
    generate_len = len(positive)

    pairs_id = np.random.randint(0, len(prot2st), [generate_len + 1000, 2])
    id2uniprot = prot2st.reset_index()['index']
    pairs_name = id2uniprot[pairs_id[:, 0]].to_numpy(
    ) + concater + id2uniprot[pairs_id[:, 1]].to_numpy()
    pairs_name = pd.Series(pairs_name)
    print(f'{len(pairs_name)} Random pair generated')

    print('Filtering by BioGRID dastaset')
    pairs_name = pairs_name[~pairs_name.isin(biogrid)]
    print(f"{len(pairs_name)} negative pair left filtering")
    pairs_name = pairs_name[:generate_len]
    print(f"{generate_len} pair selected")
    negative_names = pairs_name.str.split(concater, expand=True)

    print('Generate dataset for ppi prediction')
    namer = {0: 'a', 1: 'b'}
    pdb_names = [prot2st.loc[negative_names[i]].reset_index(drop=True).rename(
        columns={
            0: f'uniprot_{namer[i]}', 
            1: f'pdb_{namer[i]}', 
            2: f'chain_{namer[i]}',
            3: f'seq_{namer[i]}_len', 
        }) for i in range(2)
    ]
    allds = pd.concat(pdb_names, axis=1)
    # jname	label	uniprot_a	uniprot_b	seq_a_len	seq_b_len	ncontacts	pdb_a	pdb_b	pdb_gt	chain_a	chain_b
    allds.insert(0, 'jname', allds['uniprot_a'] + '-' + allds['uniprot_b'])
    allds.insert(1, 'label', 0)
    allds.insert(3, 'ncontacts', 0)
    allds.insert(4, 'pdb_gt', '-')

    finalds = pd.concat([positive, allds]).reset_index(drop=True)
    finalds.to_csv(f'datasets/{args.split}.csv')
