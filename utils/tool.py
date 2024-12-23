import torch
import yaml
import os

import numpy as np
import pandas as pd

from lightning.pytorch import seed_everything
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from biopandas.pdb import PandasPdb
from Bio.PDB import MMCIFParser

from .residues import three2oneLetter, oneHot

import warnings
warnings.filterwarnings("ignore")

def getConfig(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    seed_everything(cfg["seed"], workers=True)
    cfg['basic']['num_features'] = {
        'Rostlab/prot_bert': 1024, 
        'Rostlab/prot_t5_xl_uniref50': 1024, 
        'esm2_t33_650M_UR50D': 1280, 
        'onehot': 21
    }[cfg['basic']['embedding']]
    return cfg


def read_cif(fpath):
    parser = MMCIFParser()
    structure = parser.get_structure("", fpath)
    records = []
    for chain in structure.get_chains():
        for atom in chain.get_atoms():
            records.append([atom.name, atom.get_parent().get_resname(), chain.get_id(), *atom.coord, atom.bfactor])
    df = pd.DataFrame(records, columns=['atom', 'residue_name', 'chain_id', 'x_coord', 'y_coord', 'z_coord', 'plddt'])
    df = df[df['atom'] == 'CA']
    df = df.reset_index(drop=True)
    return df


def read_pdb(fpath):
    ppdb = PandasPdb()
    ppdb.read_pdb(fpath)
    df = ppdb.df['ATOM']
    df = df[df['atom_name'] == 'CA']
    df = df.reset_index(drop=True)
    return df


def extractPDB(pdb_path, chains=None):
    if os.path.splitext(pdb_path)[-1] == '.cif':
        CAs = read_cif(pdb_path)
    else:
        CAs = read_pdb(pdb_path)
    cod = ['x_coord', 'y_coord', 'z_coord']

    if chains is None:
        sequence = CAs['residue_name'].to_list()
        sequence = formatPDBSeq(sequence, '')
        coords = CAs[cod].to_numpy()
    
    elif chains=='auto':
        chains = CAs['chain_id'].unique()
        sequence = [CAs[CAs['chain_id'] == c]['residue_name'].to_list() for c in chains]
        sequence = [formatPDBSeq(c, '') for c in sequence]
        coords = [CAs[CAs['chain_id'] == c][cod].to_numpy() for c in chains]

    elif isinstance(chains, list):
        sequence = [CAs[CAs['chain_id'] == c]['residue_name'].to_list() for c in chains]
        sequence = [formatPDBSeq(c, '') for c in sequence]
        coords = [CAs[CAs['chain_id'] == c][cod].to_numpy() for c in chains]

    else:
        sequence = CAs[CAs['chain_id'] == chains]['residue_name'].to_list()
        sequence = formatPDBSeq(sequence, '')
        coords = CAs[CAs['chain_id'] == chains][cod].to_numpy()
    
    return sequence, coords


def formatPDBSeq(sequence, delimiter=''):
    return delimiter.join([three2oneLetter.get(i, 'X') for i in sequence])


def formatOneChars(sequence, delimiter=' '):
    return delimiter.join(sequence)


class Embed:
    featureLen = 1024
    
    def encode_prot(self, sequence):
        ids = self.tokenizer(formatOneChars(sequence), return_tensors="pt")
        with torch.no_grad():
            embedding = self.model(
                input_ids=ids['input_ids'].to(self.model.device), 
                attention_mask=ids['attention_mask'].to(self.model.device)
            )
            embedding = embedding.last_hidden_state.cpu()[0, :len(sequence)]
        return embedding

    def encode_esm(self, sequence, attention_contact=False):
        seqlen = len(sequence)
        data = [("protein1", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        if attention_contact:
            return token_representations[0, 1:seqlen+1], results["contacts"][0, :seqlen, :seqlen]
        else:
            return token_representations[0, 1:seqlen+1]

    
    def encode_onehot(self, sequence):
        x = torch.tensor([oneHot[i] for i in sequence], dtype=int)
        hot = torch.nn.functional.one_hot(x , num_classes=len(oneHot))
        return hot.type(torch.float32)


    def __init__(
            self, 
            embedding: str = "Rostlab/prot_bert",
            device='cpu'
        ):
        ALLOWED_MODELS = ['Rostlab/prot_bert', 'Rostlab/prot_t5_xl_uniref50', 'esm2_t33_650M_UR50D', 'onehot']
        assert embedding in ALLOWED_MODELS, f"This function is only built for {ALLOWED_MODELS}, get {embedding}"

        self.device = device

        print('Initializing protein sequence embedder ... ', end='')

        if embedding == 'Rostlab/prot_bert':
            self.featureLen = 1024
            self.tokenizer = BertTokenizer.from_pretrained(embedding, do_lower_case=False)
            self.model = BertModel.from_pretrained(embedding)
            self.model = self.model.to(device)
            self.model = self.model.eval()
            self.encode = self.encode_prot

        elif embedding == 'Rostlab/prot_t5_xl_uniref50':
            self.featureLen = 1024
            self.tokenizer = T5Tokenizer.from_pretrained(embedding, do_lower_case=False )
            self.model = T5EncoderModel.from_pretrained(embedding)
            self.model = self.model.to(device)
            self.model = self.model.eval()
            self.encode = self.encode_prot

        elif embedding == 'esm2_t33_650M_UR50D':
            self.featureLen = 1280
            self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", embedding)
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.to(device)
            self.model.eval()
            self.encode = self.encode_esm
        
        elif embedding == 'onehot':
            self.featureLen = 21
            self.encode = self.encode_onehot

        print('Done')
