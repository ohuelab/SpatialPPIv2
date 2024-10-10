import os
import argparse

import json
import torch

from utils.model import InteractionPredictor
from utils.tool import getConfig, extractAFPred, pdb2data, Embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="./config/default.yaml", help="The config file to use")
    parser.add_argument(
        "--cif", help="PDB path for interactor A", required=True)
    args = parser.parse_args()

    cfg = getConfig(args.config)
    device = torch.device("cuda" if (cfg['use_cuda'] and torch.cuda.is_available(
    ) and torch.cuda.device_count() > 0) else "cpu")

    # load pdbs
    seqs, coords, nc, plddt = extractAFPred(
        args.cif, cfg['InteractionPredictor']['dataset']['distance_threshold'])

    # load model
    model = InteractionPredictor.load_from_checkpoint(
        cfg['InteractionPredictor']['model']['checkpoint'],
        cfg=cfg['InteractionPredictor']).to(device)
    embedder = Embed(cfg['InteractionPredictor']['dataset']['embedding'])

    # generate feature
    input_data = pdb2data(
        seqs,
        coords,
        cfg['InteractionPredictor']['dataset']['distance_threshold'],
        embedder,
        usage='no_gt'
    ).to(model.device)
    prob = model(
        input_data.x,
        input_data.edge_index,
        input_data.edge_attr,
        [input_data.data_shape]
    )

    prob = prob.detach().cpu().tolist()

    print('Interaction Probability:', round(prob[0], 4))
    print('AF Pred Contacted Residue:', nc)
    print('AF Pred Average Interface pLDDT:', round(plddt, 2))
