import argparse
import csv
from pathlib import Path

import torch

from utils.dataset import build_data, build_data_from_adj
from utils.model import getModel
from utils.tool import Embed, extractPDB, getConfig, read_fasta


FASTA_EXTENSIONS = {".fasta", ".fa", ".faa", ".fna"}
STRUCTURE_EXTENSIONS = {".pdb", ".cif"}


def is_fasta_file(path: Path) -> bool:
    return path.suffix.lower() in FASTA_EXTENSIONS


def is_structure_file(path: Path) -> bool:
    return path.suffix.lower() in STRUCTURE_EXTENSIONS


def is_supported_file(path: Path) -> bool:
    return is_fasta_file(path) or is_structure_file(path)


def load_query_features(query_path: Path, chain: str, model_type: str, embedder: Embed):
    if model_type == "ESM-2+ac":
        seq = read_fasta(str(query_path))
        feature, adj = embedder.encode(seq, attention_contact=True)
        return {"feature": feature, "adj": adj}

    seq, coords = extractPDB(str(query_path), chain)
    feature = embedder.encode(seq)
    return {"feature": feature, "coords": coords}


def infer_pair(
    model,
    device,
    model_type: str,
    query_cached: dict,
    target_path: Path,
    query_name: str,
    chain_target: str,
    embedder: Embed,
):
    if model_type == "ESM-2+ac":
        seq_b = read_fasta(str(target_path))
        feat_b, adj_b = embedder.encode(seq_b, attention_contact=True)
        input_data = build_data_from_adj(
            features=[query_cached["feature"], feat_b],
            adjs=[query_cached["adj"], adj_b],
        ).to(device)
    else:
        seq_b, coord_b = extractPDB(str(target_path), chain_target)
        feat_b = embedder.encode(seq_b)
        input_data = build_data(
            node_feature=torch.cat([query_cached["feature"], feat_b]),
            coords=[query_cached["coords"], coord_b],
        ).to(device)

    with torch.no_grad():
        probability = model(input_data).cpu().tolist()[0]

    return {
        "query": query_name,
        "target": target_path.name,
        "probability": probability,
        "status": "ok",
        "error": "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        help="Path to query protein file (.fasta/.fa/.faa/.fna or .pdb/.cif).",
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Path to folder containing target protein files.",
    )
    parser.add_argument(
        "--output",
        default="inference_batch.csv",
        help="Output CSV path. Default: inference_batch.csv",
    )
    parser.add_argument(
        "--chain_query",
        default="first",
        help="Chain ID for query structure file. Ignored for FASTA. Default: first",
    )
    parser.add_argument(
        "--chain_targets",
        default="first",
        help="Chain ID for target structure files. Ignored for FASTA. Default: first",
    )
    parser.add_argument(
        "--model",
        default="ProtT5",
        choices=["ProtT5", "ESM-2+ac"],
        help="Model for structure inputs. FASTA inputs always force ESM-2+ac.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (e.g., cuda or cpu). Default: cuda",
    )
    args = parser.parse_args()

    query_path = Path(args.query)
    target_dir = Path(args.targets)
    output_path = Path(args.output)

    if not query_path.is_file():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    if not is_supported_file(query_path):
        raise ValueError(f"Unsupported query file type: {query_path.suffix}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target folder not found: {target_dir}")

    target_files = sorted([p for p in target_dir.iterdir() if p.is_file() and is_supported_file(p)])
    if not target_files:
        raise ValueError(f"No supported files found in target folder: {target_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    query_is_fasta = is_fasta_file(query_path)
    targets_are_fasta = [is_fasta_file(p) for p in target_files]

    if query_is_fasta:
        if not all(targets_are_fasta):
            raise ValueError("Query is FASTA, so all target files must also be FASTA.")
        model_type = "ESM-2+ac"
    else:
        if any(targets_are_fasta):
            raise ValueError("Query is structure file, so all target files must be structure files (.pdb/.cif).")
        model_type = args.model

    print("Selected model:", model_type)
    print("Loading embedder. Language model weights may download on first run.")
    if model_type == "ESM-2+ac":
        ckpt = "checkpoint/SpatialPPIv2_ESM.ckpt"
        embedder = Embed("esm2_t33_650M_UR50D", device)
    else:
        ckpt = "checkpoint/SpatialPPIv2_ProtT5.ckpt"
        embedder = Embed("Rostlab/prot_t5_xl_uniref50", device)

    cfg = getConfig("config/default.yaml")
    cfg["basic"]["num_features"] = embedder.featureLen

    model = getModel(cfg, ckpt=ckpt).to(device)
    model.eval()

    print("Encoding query protein once for batch screening...")
    query_cached = load_query_features(query_path, args.chain_query, model_type, embedder)

    results = []
    for target in target_files:
        try:
            rec = infer_pair(
                model=model,
                device=device,
                model_type=model_type,
                query_cached=query_cached,
                target_path=target,
                query_name=query_path.name,
                chain_target=args.chain_targets,
                embedder=embedder,
            )
            print(f"{target.name}: {rec['probability']:.6f}")
            results.append(rec)
        except Exception as exc:
            print(f"{target.name}: failed ({exc})")
            results.append(
                {
                    "query": query_path.name,
                    "target": target.name,
                    "probability": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "target", "probability", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} results to {output_path}")
