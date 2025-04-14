import os
import argparse
import json
import torch
import gc

from utils.model import getModel
from utils.dataset import build_data, build_data_from_adj
from utils.tool import getConfig, extractPDB, read_fasta, Embed

def run_interaction_prediction(A, B, chain_A='first', chain_B='first', model='ProtT5', device='cuda'):
    # ---- check if device available
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ----- check input file format
    if os.path.splitext(A)[-1] == '.fasta' or os.path.splitext(B)[-1] == '.fasta':
        assert os.path.splitext(A)[-1] == '.fasta' and os.path.splitext(B)[-1] == '.fasta', \
            "If using ESM-2+ac model, all input files must be in fasta format"
        print('Input file in fasta format, using ESM-2+ac model')
        print('Notice: This model is less accurate than protein structure based model.')
        model_type = 'ESM-2+ac'
    else:
        model_type = model
    
    print('Loading embedder. Notice: The script will download the language model on the first run. This may take some time.')
    if model_type == 'ESM-2+ac':
        ckpt = 'checkpoint/SpatialPPIv2_ESM.ckpt'
        embedder = Embed('esm2_t33_650M_UR50D', device)
    elif model_type == 'ProtT5':
        ckpt = 'checkpoint/SpatialPPIv2_ProtT5.ckpt'
        embedder = Embed('Rostlab/prot_t5_xl_uniref50', device)
    
    # Get default config
    cfg = getConfig(f'{os.path.dirname(__file__)}/config/default.yaml')
    cfg['basic']['num_features'] = embedder.featureLen
    
    # Load model
    model = getModel(cfg, ckpt=ckpt).to(device)
    model.eval()
    
    # Load data
    if model_type == 'ESM-2+ac':
        seq_a = read_fasta(A)
        seq_b = read_fasta(B)
        seq_a, adj_a = embedder.encode(seq_a, attention_contact=True)
        seq_b, adj_b = embedder.encode(seq_b, attention_contact=True)
        input_data = build_data_from_adj(
            features=[seq_a, seq_b],
            adjs=[adj_a, adj_b]
        ).to(device)
    elif model_type == 'ProtT5':
        seq_a, coord_a = extractPDB(A, chain_A)
        seq_b, coord_b = extractPDB(B, chain_B)
        
        input_data = build_data(
            node_feature=torch.cat([embedder.encode(i) for i in [seq_a, seq_b]]),
            coords=[coord_a, coord_b],
        ).to(device)
    
    print('Input data loaded, shape:', input_data.data_shape)
    
    with torch.no_grad():
        output = model(input_data).cpu().tolist()[0]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print('Possibility of interaction:', output)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", required=True, help="Input protein A. PDB/cif or fasta file if using ESM-2+ac model.")
    parser.add_argument("--chain_A", default='first', help="Chain ID for input A. Applies only for PDB/cif files.")
    parser.add_argument("--B", required=True, help="Input protein B. PDB/cif or fasta file if using ESM-2+ac model.")
    parser.add_argument("--chain_B", default='first', help="Chain ID for input B. Applies only for PDB/cif files.")
    parser.add_argument("--model", default="ProtT5", choices=['ProtT5', 'ESM-2+ac'], help="Model to use.")
    parser.add_argument("--device", default='cuda', help="Device to use (default: cuda).")
    args = parser.parse_args()
    
    run_interaction_prediction(args.A, args.B, args.chain_A, args.chain_B, args.model, args.device)

if __name__ == "__main__":
    main()
