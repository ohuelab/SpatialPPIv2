import os
import argparse

import json
import torch

from utils.model import SpatialPPI2
from utils.tool import getConfig, extractPDB, pdb2data, Embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="./config/default.yaml", help="The config file to use")
    parser.add_argument(
        "--pdb_a", help="PDB path for interactor A", required=True)
    parser.add_argument(
        "--pdb_b", help="PDB path for interactor B", required=True)
    parser.add_argument("--output", help="Output json path", required=True)
    args = parser.parse_args()

    cfg = getConfig(args.config)
    device = torch.device("cuda" if (cfg['use_cuda'] and torch.cuda.is_available(
    ) and torch.cuda.device_count() > 0) else "cpu")

    # load pdbs
    seq_a, coord_a = extractPDB(args.pdb_a)
    seq_b, coord_b = extractPDB(args.pdb_b)

    # load model
    model = SpatialPPI2(cfg).to(device)
    embedder = Embed(cfg['InteractionPredictor']['dataset']['embedding'])

    # generate feature
    input_data = pdb2data(
        [seq_a, seq_b],
        [coord_a, coord_b],
        cfg['InteractionPredictor']['dataset']['distance_threshold'],
        embedder,
        usage='no_gt'
    ).to(model.device)
    prob, contact = model(
        input_data.x,
        input_data.edge_index,
        input_data.edge_attr,
        [input_data.data_shape]
    )

    prob = prob.detach().cpu().tolist()
    contact = contact.detach().cpu().tolist()

    with open(args.output, 'w') as f:
        f.write(json.dumps({
            'Interaction Probability': prob,
            'Contact Map': contact
        }))
