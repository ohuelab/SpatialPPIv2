import os
import sys
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance_matrix

from pinder.core import get_index

sys.path.append('.')
from utils.tool import extractPDB, getConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude", help="Dataset to exclude")
    parser.add_argument("--biogrid", help="Path to biogrid files")
    args = parser.parse_args()

    # Load configs
    cfg = getConfig(args.config)

    index = get_index().copy()
    index = index[index['split'] == args.split]

    pdblist = index['id'].to_numpy() + '.pdb'
    data_root = cfg['dataset'][args.split]['data_root']
    distance_threshold = cfg['basic']['distance_threshold']

    # filter pinder dataset
    print('Filtering pinder dataset')

    seq_as = []
    seq_bs = []
    n_contacts = []
    for fpath in tqdm(pdblist):
        pdb_path = os.path.join(data_root, fpath)
        if os.path.exists(pdb_path):
            (seq_a, seq_b), (coord_a, coord_b) = extractPDB(pdb_path, ['L', 'R'])
            contacts = np.count_nonzero(distance_matrix(coord_a, coord_b) < distance_threshold)
            seq_as.append(seq_a)
            seq_bs.append(seq_b)
            n_contacts.append(contacts)
        else:
            seq_as.append('')
            seq_bs.append('')
            n_contacts.append(0)

    positive = pd.DataFrame({
        'jname': index['uniprot_L'].to_numpy() + '-' + index['uniprot_R'].to_numpy(),
        'label': np.ones(len(pdblist), dtype=int),
        'uniprot_a': index['uniprot_L'].to_numpy(),
        'uniprot_b': index['uniprot_R'].to_numpy(),
        'seq_a': seq_as,
        'seq_b': seq_bs,
        'ncontacts': n_contacts,
        'pdb_a': pdblist,
        'pdb_b': pdblist,
        'pdb_gt': pdblist,
        'chain_a': ['L'] * len(pdblist),
        'chain_b': ['R'] * len(pdblist)
    })

    positive = positive[(positive['seq_a'].str.len() > 35) &
                        (positive['seq_a'].str.len() < 300)]
    positive = positive[(positive['seq_b'].str.len() > 35) &
                        (positive['seq_b'].str.len() < 300)]
    positive = positive[positive['ncontacts'] > 8]

    if args.exclude is not None:
        print('Removing exclude')
        exclude = pd.read_csv(args.exclude, header=None)[0]
        ind = positive['uniprot_a'].isin(exclude) | positive['uniprot_a'].isin(exclude)
        positive = positive[~ind]

    uniprot_a = positive.drop_duplicates('uniprot_a', keep='first').get(
        ['uniprot_a', 'pdb_gt', 'chain_a', 'seq_a']).copy().rename(
            columns={'uniprot_a': 'uniprot', 'chain_a': 'chain', 'seq_a': 'seq'})
    uniprot_b = positive.drop_duplicates('uniprot_b', keep='first').get(
        ['uniprot_b', 'pdb_gt', 'chain_b', 'seq_b']).copy().rename(
            columns={'uniprot_b': 'uniprot', 'chain_b': 'chain', 'seq_b': 'seq'})
    uniprot = pd.concat([uniprot_a, uniprot_b]).drop_duplicates(
        'uniprot', keep='first')
    prot2st = pd.DataFrame(uniprot.get(
        ['uniprot', 'pdb_gt', 'chain' ,'seq']).to_numpy(), index=uniprot['uniprot'].to_numpy())
    prot2st = prot2st[~prot2st[1].str.contains('UNDEFINED')]
    concater = '^'
    # Generate Negative Samples
    print('Generating negative samples')
    generate_len = len(positive)

    pairs_id = np.random.randint(0, len(prot2st), [generate_len + 1000, 2])
    id2uniprot = prot2st.reset_index()['index']
    pairs_name = id2uniprot[pairs_id[:, 0]].to_numpy(
    ) + concater + id2uniprot[pairs_id[:, 1]].to_numpy()
    pairs_name = pd.Series(pairs_name)
    print(f'{len(pairs_name)} Random pair generated')

    if args.biogrid is not None:
        print('Load BioGRID database')
        biogrid_ds = pd.read_csv(args.biogrid, delimiter='\t')
        protkey_a = 'SWISS-PROT Accessions Interactor A'
        protkey_b = 'SWISS-PROT Accessions Interactor B'
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
            3: f'seq_{namer[i]}', 
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
