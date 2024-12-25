import os
import argparse

import json
import torch

from utils.model import getModel
from utils.dataset import build_data, build_data_from_adj
from utils.tool import getConfig, extractPDB, read_fasta, Embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--A", 
        help="Input protein A. Should be a PDB/cif file or fasta file if use ESM-2+ac model.", 
        required=True
    )
    parser.add_argument(
        "--chain_A",
        default='first',
        help="The chain ID for input A. Only apply when the input is PDB/cif file. default will use the first chain in the file.", 
    )
    parser.add_argument(
        "--B", 
        help="Input protein B. Should be a PDB/cif file or fasta file if use ESM-2+ac model.", 
        required=True
    )
    parser.add_argument(
        "--chain_B",
        default='first',
        help="The chain ID for input B. Only apply when the input is PDB/cif file. default will use the first chain in the file.", 
    )
    parser.add_argument(
        "--model", 
        default="ProtT5",
        choices=['ProtT5', 'ESM-2+ac'],
        help="Which model to use. If the input file in fasta format, use ESM-2+ac."
    )
    parser.add_argument(
        "--device", 
        default='cuda',
        help="The device to use. Default use cuda."
    )
    args = parser.parse_args()

    # ---- check if device available
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ----- check input file format
    if os.path.splitext(args.A)[-1] == '.fasta' or os.path.splitext(args.B)[-1] == '.fasta':
        assert os.path.splitext(args.A)[-1] == '.fasta' and os.path.splitext(args.B)[-1] == '.fasta', "If use ESM-2+ac model, all input file must be fasta format"
        print('Input file in fasta format, use ESM-2+ac model')
        print('Notice: This model is less accurate than protein structure based model.')
        model_type = 'ESM-2+ac'
    else:
        model_type = args.model
    
    print('Loading embedder. Notice: The script will download the language model at the first run. This may take some time.')
    if model_type == 'ESM-2+ac':
        ckpt = 'checkpoint/SpatialPPIv2_ESM.ckpt'
        embedder = Embed('esm2_t33_650M_UR50D', device)
    elif model_type == 'ProtT5':
        ckpt = 'checkpoint/SpatialPPIv2_ProtT5.ckpt'
        embedder = Embed('Rostlab/prot_t5_xl_uniref50', device)

    # get default config
    cfg = getConfig('config/default.yaml')
    cfg['basic']['num_features'] = embedder.featureLen

    # load model
    model = getModel(cfg, ckpt=ckpt).to(device)
    model.eval()

    # load data
    if model_type == 'ESM-2+ac':
        seq_a = read_fasta(args.A)
        seq_b = read_fasta(args.B)
        seq_a, adj_a = embedder.encode(seq_a, attention_contact=True)
        seq_b, adj_b = embedder.encode(seq_b, attention_contact=True)
        input_data = build_data_from_adj(
            features=[seq_a, seq_b],
            adjs=[adj_a, adj_b]
        ).to(device)
    elif model_type == 'ProtT5':
        seq_a, coord_a = extractPDB(args.A, args.chain_A)
        seq_b, coord_b = extractPDB(args.B, args.chain_B)

        input_data = build_data(
            node_feature=torch.cat([embedder.encode(i) for i in [seq_a, seq_b]]), 
            coords=[coord_a, coord_b], 
        ).to(device)

    print('Input data loaded, shape:', input_data.data_shape)

    with torch.no_grad():
        output = model(input_data).cpu().tolist()[0]
    
    print('Possibility of interaction:', output)
