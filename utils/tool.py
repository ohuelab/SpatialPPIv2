import torch
import yaml
import numpy as np
import pandas as pd

from lightning.pytorch import seed_everything
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
from Bio.PDB import MMCIFParser

from biopandas.pdb import PandasPdb

from .residues import three2oneLetter

import warnings
warnings.filterwarnings("ignore")

def getConfig(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    seed_everything(cfg["seed"], workers=True)
    return cfg


def get_distance_matrix(coords):
    diff_tensor = torch.unsqueeze(
        coords, axis=1) - torch.unsqueeze(coords, axis=0)
    distance_matrix = torch.sqrt(torch.sum(torch.pow(diff_tensor, 2), axis=-1))
    return distance_matrix


def extractPDB(pdb_path, chains=None):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    df = ppdb.df['ATOM']
    CAs = df[df['atom_name'] == 'CA']
    cod = ['x_coord', 'y_coord', 'z_coord']

    if chains is None:
        sequence = CAs['residue_name'].to_list()
        coords = CAs[cod].to_numpy()
    
    elif chains=='auto':
        chains = CAs['chain_id'].unique()
        sequence = [CAs[CAs['chain_id'] == c]['residue_name'].to_list() for c in chains]
        coords = [CAs[CAs['chain_id'] == c][cod].to_numpy() for c in chains]

    elif isinstance(chains, list):
        sequence = [CAs[CAs['chain_id'] == c]['residue_name'].to_list() for c in chains]
        coords = [CAs[CAs['chain_id'] == c][cod].to_numpy() for c in chains]

    else:
        sequence = CAs[CAs['chain_id'] == chains]['residue_name'].to_list()
        coords = CAs[CAs['chain_id'] == chains][cod].to_numpy()

    return sequence, coords


def pdb2data(
    seqs,
    coords,
    distance_threshold,
    embeder,
    interact=0,
    usage=None
):
    assert len(seqs) == 2
    assert len(coords) == 2
    seq_a, seq_b = seqs
    coord_a, coord_b = coords
    assert len(seq_a) == len(coord_a)
    assert len(seq_b) == len(coord_b)
    lenA = len(coord_a)

    distance_matrix = get_distance_matrix(
        torch.cat([torch.tensor(coord_a), torch.tensor(coord_b)])
    )

    x_coords, y_coords = torch.meshgrid(
        torch.arange(distance_matrix.shape[0]),
        torch.arange(distance_matrix.shape[1]),
        indexing='ij')
    mask = x_coords >= y_coords
    distance_matrix[mask] = 0

    if usage=='no_gt':
        label = torch.zeros([len(seq_a), len(seq_b)])
    else:
        label = distance_matrix[:lenA, lenA:].clone().flatten()

    distance_matrix[:lenA, lenA:] = 0
    distance_matrix[lenA:, :lenA] = 0
    adj = (distance_threshold > distance_matrix) & (distance_matrix > 0)

    data = Data(
        x=torch.cat([embeder.encode(s) for s in [seq_a, seq_b]]),
        data_shape=[len(coord_a), len(coord_b)],
        edge_index=torch.nonzero(adj).T,
        edge_attr=distance_matrix[adj].type(torch.float32),
        interact=interact,
        y=label.type(torch.float32),
    )
    return data


def extractAFPred(cif_path, distance_threshold=8):
    parser = MMCIFParser()
    structure = parser.get_structure('af_pred', cif_path)

    records = []
    for chain in structure.get_chains():
        for atom in chain.get_atoms():
            records.append([
                atom.name, 
                atom.get_parent().get_resname(), 
                *atom.coord, 
                chain.get_id(), 
                atom.bfactor
                ])
    df = pd.DataFrame(records, columns=['atom', 'residue', 'x', 'y', 'z', 'chain', 'plddt'])
    df = df[df['atom'] == 'CA'] 
    coord_a = df[df['chain'] == 'A'][['x', 'y', 'z']].astype(float).to_numpy()
    coord_b = df[df['chain'] == 'B'][['x', 'y', 'z']].astype(float).to_numpy()
    seq_a = df[df['chain'] == 'A']['residue'].to_list()
    seq_b = df[df['chain'] == 'B']['residue'].to_list()
    contacts_pred = distance_matrix(coord_a, coord_b) < distance_threshold

    contacted_res = np.nonzero(contacts_pred)
    plddts = df['plddt'].astype(float).to_numpy()
    res_unique = [np.unique(i) for i in contacted_res]
    n_contact = len(res_unique[0]) + len(res_unique[1])
    avg_plddt = np.concatenate([plddts[res_unique[0]], plddts[res_unique[1] + len(coord_a)]])
    avg_plddt = np.average(avg_plddt) if len(avg_plddt) else 0

    return [seq_a, seq_b], [coord_a, coord_b], n_contact, avg_plddt


def formatPDBSeq(sequence):
    return " ".join([three2oneLetter.get(i, 'X') for i in sequence])


def formatOneChars(sequence):
    return " ".join(sequence)


class Embed:
    featureLen = 1024

    def __init__(
            self, 
            embedding: str = "Rostlab/prot_bert"
        ):
        self.tokenizer = BertTokenizer.from_pretrained(
            embedding, do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")

    def encode(self, sequence, formatter=formatOneChars):
        seq = formatter(sequence)
        encoded_input = self.tokenizer(seq, return_tensors="pt")
        output = self.model(**encoded_input)
        return output.last_hidden_state[0, : len(sequence)].detach()
