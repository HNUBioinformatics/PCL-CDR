from pathlib import Path
import random
from typing import Tuple, List
import warnings
import pickle
from rdkit import Chem
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import MultiViewModel

warnings.filterwarnings("ignore")
SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DP_EDGE_PATH = Path("data/drug_pathway_edges.csv")
DT_WIDE_PATH = Path("data/drug_target_edges.csv")
OMICS_PATH = Path("data/cell_omics.csv")
SSGSEA_PATH = Path("data/cell_pathway_ssgsea.csv")
DRUG_SMILES_PATH = Path("data/drug_smiles.csv")
DMPNN_HIDDEN = 256
DMPNN_DEPTH = 3
TARGET_ESM2_PATH = Path("data/target_esm2_embeddings.csv")
IC50_PATH = Path("data/drug_response.csv")
PATHWAY_GENE_MEMBERSHIP_PATH = Path("data/pathway_gene_membership.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "best_model_moe_gate.pt"
Y_SCALER_PATH = OUT_DIR / "y_scaler_moe_gate.pkl"
ATTN_PATH = OUT_DIR / "attn_maps_moe_gate_best_test.pkl"
SAVE_ATTN = True
EARLY_STOPPING = True
PATIENCE = 20
MIN_DELTA = 1e-6
PC_TOPK_PER_CELL = 50
HIDDEN = 512
EMB_DIM = 256
DROPOUT = 0.2
EPOCHS = 200
LR = 2e-4
WEIGHT_DECAY = 1e-5
PAIR_BATCH_SIZE = 128
NUM_WORKERS = 0
USE_SCHEDULER = True
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 10
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def open_with_fallback_encodings(path: Path, encodings=("utf-8-sig", "utf-8", "gbk", "cp936", "latin1")):
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(2048)
            return enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"All encodings failed for file: {path}")

def norm_name(x):
    return " ".join(str(x).strip().split()) if x else ""

def norm_gene(x):
    return " ".join(str(x).strip().split()).upper() if x else ""

def safe_corr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def spearman_corr(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return safe_corr(ra, rb)

def regression_metrics(y_true, y_pred):
    y_true, y_pred = y_true.astype(np.float64), y_pred.astype(np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2,
        "PCC": safe_corr(y_true, y_pred),
        "SCC": spearman_corr(y_true, y_pred),
    }

def print_metrics(prefix, m):
    print(
        f"{prefix}  MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
        f"R2={m['R2']:.4f}  PCC={m['PCC']:.4f}  SCC={m['SCC']:.4f}"
    )

def fit_y_scaler(y_train_raw):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(y_train_raw.reshape(-1, 1))
    return scaler

def transform_y(scaler, y_raw):
    return scaler.transform(y_raw.reshape(-1, 1)).reshape(-1)

def inverse_y(scaler, y_z):
    return scaler.inverse_transform(y_z.reshape(-1, 1)).reshape(-1)

def to_np(x):
    return x.detach().cpu().numpy()

def fit_transform_df_by_train_rows(df: pd.DataFrame, train_rows: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:


    train_rows = [r for r in train_rows if r in df.index]
    if len(train_rows) == 0:
        raise ValueError("train_rows is empty; check that the training split contains valid entities.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(df.loc[train_rows].to_numpy(dtype=np.float32))

    X_all = scaler.transform(df.to_numpy(dtype=np.float32))
    df_z = pd.DataFrame(X_all, index=df.index)
    return df_z, scaler


def load_ic50_long(path):
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df = df.rename(columns={df.columns[0]: "Cell_line"})
    df_long = df.melt(id_vars=["Cell_line"], var_name="drug_name", value_name="y")
    df_long["Cell_line"] = df_long["Cell_line"].map(norm_name)
    df_long["drug_name"] = df_long["drug_name"].map(norm_name)
    df_long = df_long.dropna(subset=["y"])
    df_long = df_long[~df_long["y"].astype(str).str.lower().isin(["nan", "none", ""])]
    df_long["y"] = df_long["y"].astype(float)
    return df_long

def load_drug_smiles(path) -> pd.Series:
    df = pd.read_csv(path, encoding=open_with_fallback_encodings(path))
    cols = {str(c).strip().lower().lstrip("\ufeff"): c for c in df.columns}
    drug_col = cols.get("drug_name")
    smiles_col = cols.get("smiles") or cols.get("canonical_smiles") or cols.get("isomeric_smiles")
    if drug_col is None or smiles_col is None:
        raise ValueError("The drug structure CSV requires drug_name and smiles columns.")
    if df[[drug_col, smiles_col]].isna().any().any():
        raise ValueError("Missing drug_name or smiles values; complete them before training.")
    result = {}
    for name, smi in zip(df[drug_col], df[smiles_col]):
        name, smi = norm_name(name), str(smi).strip()
        if not name or not smi:
            raise ValueError("drug_name and smiles must not be empty.")
        if name in result and result[name] != smi:
            raise ValueError(f"Conflicting SMILES for the same drug: {name}")
        result[name] = smi
    return pd.Series(result, name="smiles", dtype=object)


def one_hot_unknown(value, choices):
    return [float(value == c) for c in choices] + [float(value not in choices)]


def atom_features(atom):


    return (
        one_hot_unknown(atom.GetAtomicNum(), list(range(1, 101)))
        + one_hot_unknown(atom.GetTotalDegree(), list(range(6)))
        + one_hot_unknown(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + one_hot_unknown(int(atom.GetChiralTag()), list(range(4)))
        + one_hot_unknown(atom.GetTotalNumHs(), list(range(5)))
        + one_hot_unknown(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ])
        + [float(atom.GetIsAromatic()), atom.GetMass() * 0.01]
    )


def bond_features(bond):

    bt = bond.GetBondType()
    return [0.0] + [float(bt == b) for b in [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
    ]] + [float(bond.GetIsConjugated()), float(bond.IsInRing())] + one_hot_unknown(
        int(bond.GetStereo()), list(range(6))
    )


def build_drug_graph(smiles: pd.Series, drugs, device):


    atoms, bonds, src, dst, rev, mol_ids, counts = [], [], [], [], [], [], []
    for mol_id, drug in enumerate(drugs):
        smi = smiles.loc[drug]
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Cannot parse drug {drug!r} SMILES: {smi!r}")
        offset = len(atoms)
        n_atoms = mol.GetNumAtoms()
        atoms.extend(atom_features(a) for a in mol.GetAtoms())
        mol_ids.extend([mol_id] * n_atoms)
        counts.append(n_atoms)
        for b in mol.GetBonds():
            u, v = offset + b.GetBeginAtomIdx(), offset + b.GetEndAtomIdx()
            edge_id = len(src)
            bf = bond_features(b)
            src.extend([u, v])
            dst.extend([v, u])
            rev.extend([edge_id + 1, edge_id])
            bonds.extend([bf, bf])
    if not atoms:
        raise ValueError("No drug molecules are available for training.")
    def idx(values):
        return torch.tensor(values, dtype=torch.long, device=device)
    graph = {
        "atom_features": torch.tensor(atoms, dtype=torch.float32, device=device),
        "bond_features": torch.tensor(bonds, dtype=torch.float32, device=device).reshape(-1, 14),
        "edge_src": idx(src), "edge_dst": idx(dst), "reverse_edge": idx(rev),
        "atom_to_molecule": idx(mol_ids), "atom_counts": idx(counts),
        "num_molecules": len(drugs),
    }
    print(f"[DMPNN graph] drugs={len(drugs)} atoms={len(atoms)} directed_bonds={len(src)}")
    return graph


def load_target_esm2_raw(path) -> pd.DataFrame:
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    prot_key = cols_lower.get("protein") or cols_lower.get("portein")
    if not prot_key:
        raise ValueError("The target table is missing the protein column.")
    prots = df[prot_key].map(norm_gene).tolist()
    X = df.drop(columns=[prot_key]).astype(np.float32).to_numpy()
    print(f"[Target ESM2 RAW] {len(prots)} targets, dim={X.shape[1]}")
    return pd.DataFrame(X, index=prots)

def load_cell_omics_raw(path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    cell_col = cols_lower.get("cell_line", df.columns[0])
    df[cell_col] = df[cell_col].map(norm_name)

    mutation_cols = [c for c in df.columns if str(c).lower().endswith("_mutation")]
    cnv_cols = [c for c in df.columns if str(c).lower().endswith("_cnv")]
    rna_cols = [c for c in df.columns if str(c).lower().endswith("_rna")]

    print(f"[Cell Features RAW] mutation={len(mutation_cols)}, cnv={len(cnv_cols)}, rna={len(rna_cols)}")

    return tuple(
        pd.DataFrame(df[cols].to_numpy(dtype=np.float32), index=df[cell_col].tolist())
        for cols in (mutation_cols, cnv_cols, rna_cols)
    )


def load_ssgsea(path) -> pd.DataFrame:
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    cell_col = cols_lower["cell_line"]
    df[cell_col] = df[cell_col].map(norm_name)
    pathways = [c for c in df.columns if c != cell_col]
    mat = df[pathways].astype(np.float32)
    mat.index = df[cell_col]
    return mat

def load_dt_edges(path):
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    drug_col = cols_lower["drug_name"]

    edges = []
    for _, row in df.iterrows():
        drug = norm_name(row[drug_col])
        for c, v in row.items():
            if c == drug_col or pd.isna(v):
                continue
            t = norm_gene(v)
            if t and t.lower() not in {"nan", "none"}:
                edges.append((drug, t))
    return edges

def load_dp_edges(path):
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    df["drug_name"] = df[cols_lower["drug_name"]].map(norm_name)
    df["pathway"] = df[cols_lower["pathway"]].map(norm_name)
    df["w_dp"] = df[cols_lower["w_dp"]].astype(float)
    return df[["drug_name", "pathway", "w_dp"]]

def load_pt_edges(path):
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    pcol, gcol = cols_lower["pathway"], cols_lower["genes"]

    edges = []
    for _, r in df.iterrows():
        p = norm_name(r[pcol])
        if pd.isna(r[gcol]):
            continue
        genes = [norm_gene(x) for x in str(r[gcol]).split("|")]
        genes = [g for g in genes if g and g.lower() not in {"nan", "none"}]
        for g in list(dict.fromkeys(genes)):
            edges.append((p, g))
    return edges


def compute_pathway_stats_on_traincells(
    ssgsea_sub: pd.DataFrame, train_cells: List[str]
) -> Tuple[np.ndarray, np.ndarray]:


    train_cells = [c for c in train_cells if c in ssgsea_sub.index]
    if len(train_cells) == 0:
        raise ValueError("train_cells is empty; cannot compute pathway statistics.")

    X_train = ssgsea_sub.loc[train_cells].to_numpy(dtype=np.float32)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-9
    return mean, std

def build_pc_edges_from_ssgsea_trainstats(
    ssgsea_sub: pd.DataFrame,
    mean_train: np.ndarray,
    std_train: np.ndarray,
    topk: int,
) -> List[Tuple[str, str, float]]:


    cell_names = list(ssgsea_sub.index)

    path_names = list(ssgsea_sub.columns)
    X_all = ssgsea_sub.to_numpy(dtype=np.float32)


    Z_all = (X_all - mean_train.reshape(1, -1)) / std_train.reshape(1, -1)

    cell_index = {c: i for i, c in enumerate(ssgsea_sub.index)}

    edges = []
    for cell in cell_names:
        i = cell_index[cell]
        z_row = Z_all[i]

        if topk < len(path_names):
            idx = np.argpartition(np.abs(z_row), -topk)[-topk:]
        else:
            idx = np.arange(len(path_names))
        idx = idx[np.argsort(-np.abs(z_row[idx]))]

        for j in idx:
            p = path_names[j]
            w = float(abs(z_row[j]))
            edges.append((p, cell, w))
    return edges


class PairDataset(Dataset):
    def __init__(self, dr_idx: torch.Tensor, ce_idx: torch.Tensor, y: torch.Tensor):
        self.dr = dr_idx
        self.ce = ce_idx
        self.y = y

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, i):
        return self.dr[i], self.ce[i], self.y[i]


def train_one_epoch(
    model, optimizer, loader,
    drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
    edges_dt, edges_dp, w_dp, edges_pc, w_pc, edges_pt,
):
    model.train()
    total_loss, n_total = 0.0, 0

    for b_dr, b_ce, b_y in loader:
        b_dr = b_dr.to(DEVICE, non_blocking=True)
        b_ce = b_ce.to(DEVICE, non_blocking=True)
        b_y  = b_y.to(DEVICE, non_blocking=True)

        pred_z, _ = model(
            drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
            edges_dt,
            edges_dp, w_dp,
            edges_pc, w_pc,
            edges_pt,
            b_dr, b_ce,
        )

        loss = F.mse_loss(pred_z, b_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = b_y.numel()
        total_loss += float(loss.item()) * bs
        n_total    += bs

    return total_loss / max(n_total, 1)

@torch.no_grad()
def eval_split(
    model, loader, y_scaler,
    drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
    edges_dt, edges_dp, w_dp, edges_pc, w_pc, edges_pt,
):
    model.eval()
    preds, targs = [], []
    aux_last = None

    for b_dr, b_ce, b_y in loader:
        b_dr = b_dr.to(DEVICE, non_blocking=True)
        b_ce = b_ce.to(DEVICE, non_blocking=True)
        b_y  = b_y.to(DEVICE, non_blocking=True)

        p_z, aux = model(
            drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
            edges_dt,
            edges_dp, w_dp,
            edges_pc, w_pc,
            edges_pt,
            b_dr, b_ce,
        )
        preds.append(p_z.detach().cpu())
        targs.append(b_y.detach().cpu())
        aux_last = aux

    p_z_np = torch.cat(preds).numpy()
    y_z_np = torch.cat(targs).numpy()

    p_raw = inverse_y(y_scaler, p_z_np)
    y_raw = inverse_y(y_scaler, y_z_np)

    metrics = regression_metrics(y_raw, p_raw)
    val_loss = metrics["RMSE"]
    return val_loss, metrics, aux_last


def main():
    set_seed(SEED)
    print(f"[Device] {DEVICE}")


    df_pairs = load_ic50_long(IC50_PATH)
    print(f"[IC50] total pairs={len(df_pairs)}  (from {IC50_PATH})")


    drug_smiles = load_drug_smiles(DRUG_SMILES_PATH)
    target_feat_raw = load_target_esm2_raw(TARGET_ESM2_PATH)
    cell_mut_raw, cell_cnv_raw, cell_rna_raw = load_cell_omics_raw(OMICS_PATH)
    ssgsea_mat = load_ssgsea(SSGSEA_PATH)


    dt_edges_raw = load_dt_edges(DT_WIDE_PATH)
    dp_df = load_dp_edges(DP_EDGE_PATH)
    pt_edges_raw = load_pt_edges(PATHWAY_GENE_MEMBERSHIP_PATH)


    missing_smiles = sorted(set(df_pairs["drug_name"]) - set(drug_smiles.index))
    if missing_smiles:
        raise ValueError(f"Drugs in the response data are missing SMILES; provide them to preserve the sample set: {missing_smiles}")
    drugs = sorted(df_pairs["drug_name"].unique().tolist())
    cells = sorted(list(set(df_pairs["Cell_line"]) & set(cell_mut_raw.index) & set(cell_cnv_raw.index) & set(cell_rna_raw.index) & set(ssgsea_mat.index)))


    path_candidates = set(p for p, _ in pt_edges_raw)
    pathways = sorted(list(set(ssgsea_mat.columns) & path_candidates))


    targ_candidates = set([t for _, t in dt_edges_raw]) | set([t for _, t in pt_edges_raw])
    targets = sorted(list(set(target_feat_raw.index) & targ_candidates))

    df_pairs = df_pairs[df_pairs["drug_name"].isin(drugs) & df_pairs["Cell_line"].isin(cells)].reset_index(drop=True)
    print(f"[Align] drugs={len(drugs)} cells={len(cells)} pathways={len(pathways)} targets={len(targets)} pairs={len(df_pairs)}")


    drug2i = {d: i for i, d in enumerate(drugs)}
    cell2i = {c: i for i, c in enumerate(cells)}
    path2i = {p: i for i, p in enumerate(pathways)}
    targ2i = {t: i for i, t in enumerate(targets)}

    pair_dr = df_pairs["drug_name"].map(drug2i).to_numpy(np.int64)
    pair_ce = df_pairs["Cell_line"].map(cell2i).to_numpy(np.int64)
    y_all_raw = df_pairs["y"].to_numpy(np.float32)

    N = len(df_pairs)
    idx = np.arange(N)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    n_train, n_val = int(N * TRAIN_RATIO), int(N * VAL_RATIO)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


    train_drug_ids = np.unique(pair_dr[train_idx])
    train_cell_ids = np.unique(pair_ce[train_idx])
    train_drug_names = [drugs[i] for i in train_drug_ids.tolist()]
    train_cell_names = [cells[i] for i in train_cell_ids.tolist()]


    cell_mut_z, cell_mut_scaler = fit_transform_df_by_train_rows(cell_mut_raw.loc[cells], train_cell_names)
    cell_cnv_z, cell_cnv_scaler = fit_transform_df_by_train_rows(cell_cnv_raw.loc[cells], train_cell_names)
    cell_rna_z, cell_rna_scaler = fit_transform_df_by_train_rows(cell_rna_raw.loc[cells], train_cell_names)


    train_drug_set = set(train_drug_names)
    train_target_names = sorted({t for d, t in dt_edges_raw if d in train_drug_set and t in targets}) or targets
    target_feat_z, target_scaler = fit_transform_df_by_train_rows(target_feat_raw.loc[targets], train_target_names)


    ssgsea_sub = ssgsea_mat.loc[cells, pathways].astype(np.float32)
    mean_train, std_train = compute_pathway_stats_on_traincells(ssgsea_sub, train_cell_names)
    pc_edges_raw = build_pc_edges_from_ssgsea_trainstats(
        ssgsea_sub, mean_train, std_train, PC_TOPK_PER_CELL
    )


    y_train_raw = y_all_raw[train_idx].astype(np.float32)
    y_val_raw   = y_all_raw[val_idx].astype(np.float32)
    y_test_raw  = y_all_raw[test_idx].astype(np.float32)

    y_scaler = fit_y_scaler(y_train_raw)
    y_train_z = transform_y(y_scaler, y_train_raw).astype(np.float32)
    y_val_z   = transform_y(y_scaler, y_val_raw).astype(np.float32)
    y_test_z  = transform_y(y_scaler, y_test_raw).astype(np.float32)

    with open(Y_SCALER_PATH, "wb") as f:
        pickle.dump(y_scaler, f)
    print(f"[Saved] y scaler -> {Y_SCALER_PATH}")


    drug_graph = build_drug_graph(drug_smiles, drugs, DEVICE)
    X_cell_mut = torch.tensor(cell_mut_z.loc[cells].to_numpy(np.float32), device=DEVICE)
    X_cell_cnv = torch.tensor(cell_cnv_z.loc[cells].to_numpy(np.float32), device=DEVICE)
    X_cell_rna = torch.tensor(cell_rna_z.loc[cells].to_numpy(np.float32), device=DEVICE)
    X_targ = torch.tensor(target_feat_z.loc[targets].to_numpy(np.float32), device=DEVICE)

    print(f"[FeatDim] drug_dmpnn={EMB_DIM} target={X_targ.size(1)} mutation={X_cell_mut.size(1)} cnv={X_cell_cnv.size(1)} rna={X_cell_rna.size(1)}")


    def build_dt_edges():
        src_d, dst_t = [], []
        for d, t in dt_edges_raw:
            if d in drug2i and t in targ2i:
                src_d.append(drug2i[d])
                dst_t.append(targ2i[t])
        src_d = torch.tensor(src_d, dtype=torch.long, device=DEVICE)
        dst_t = torch.tensor(dst_t, dtype=torch.long, device=DEVICE)
        return {"target2drug": (dst_t, src_d)}


    def build_dp_edges():
        src_d, dst_p, ww = [], [], []
        for _, r in dp_df.iterrows():
            d, p, w = r["drug_name"], r["pathway"], float(r["w_dp"])
            if d in drug2i and p in path2i:
                src_d.append(drug2i[d])
                dst_p.append(path2i[p])
                ww.append(w)
        src_d = torch.tensor(src_d, dtype=torch.long, device=DEVICE)
        dst_p = torch.tensor(dst_p, dtype=torch.long, device=DEVICE)
        ww = torch.tensor(ww, dtype=torch.float32, device=DEVICE)
        edges = {"pathway2drug": (dst_p, src_d)}
        w_dict = {"pathway2drug": ww}
        return edges, w_dict


    def build_pt_edges():
        src_p, dst_t = [], []
        for p, t in pt_edges_raw:
            if p in path2i and t in targ2i:
                src_p.append(path2i[p])
                dst_t.append(targ2i[t])
        src_p = torch.tensor(src_p, dtype=torch.long, device=DEVICE)
        dst_t = torch.tensor(dst_t, dtype=torch.long, device=DEVICE)
        edges = {"pathway2target": (src_p, dst_t), "target2pathway": (dst_t, src_p)}
        return edges


    def build_pc_edges(pc_edges_raw_list: List[Tuple[str, str, float]]):
        src_p, dst_c, ww = [], [], []
        for p, c, w in pc_edges_raw_list:
            if p in path2i and c in cell2i:
                src_p.append(path2i[p])
                dst_c.append(cell2i[c])
                ww.append(float(w))
        src_p = torch.tensor(src_p, dtype=torch.long, device=DEVICE)
        dst_c = torch.tensor(dst_c, dtype=torch.long, device=DEVICE)
        ww = torch.tensor(ww, dtype=torch.float32, device=DEVICE)
        edges = {"pathway2cell": (src_p, dst_c)}
        w_dict = {"pathway2cell": ww}
        return edges, w_dict

    edges_dt = build_dt_edges()
    edges_dp, w_dp = build_dp_edges()
    edges_pt = build_pt_edges()
    edges_pc, w_pc = build_pc_edges(pc_edges_raw)


    train_dr = torch.tensor(pair_dr[train_idx], dtype=torch.long)
    train_ce = torch.tensor(pair_ce[train_idx], dtype=torch.long)
    train_y  = torch.tensor(y_train_z, dtype=torch.float32)

    val_dr = torch.tensor(pair_dr[val_idx], dtype=torch.long)
    val_ce = torch.tensor(pair_ce[val_idx], dtype=torch.long)
    val_y  = torch.tensor(y_val_z, dtype=torch.float32)

    test_dr = torch.tensor(pair_dr[test_idx], dtype=torch.long)
    test_ce = torch.tensor(pair_ce[test_idx], dtype=torch.long)
    test_y  = torch.tensor(y_test_z, dtype=torch.float32)

    train_loader = DataLoader(PairDataset(train_dr, train_ce, train_y),
                              batch_size=PAIR_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=False)
    val_loader   = DataLoader(PairDataset(val_dr, val_ce, val_y),
                              batch_size=PAIR_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, drop_last=False)
    test_loader  = DataLoader(PairDataset(test_dr, test_ce, test_y),
                              batch_size=PAIR_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, drop_last=False)


    model = MultiViewModel(
        atom_dim=drug_graph["atom_features"].size(1),
        bond_dim=drug_graph["bond_features"].size(1),
        dmpnn_hidden=DMPNN_HIDDEN,
        dmpnn_depth=DMPNN_DEPTH,
        in_target=X_targ.size(1),
        num_pathways=len(pathways),
        in_cell_mut=X_cell_mut.size(1),
        in_cell_cnv=X_cell_cnv.size(1),
        in_cell_rna=X_cell_rna.size(1),
        hidden=HIDDEN,
        emb_dim=EMB_DIM,
        dropout=DROPOUT,
        num_heads=4,
        concat=True,
        gate_hidden=None,
    ).to(DEVICE)


    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
        )

    best_val, patience_cnt, best_epoch = 1e18, 0, 0


    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model, optim, train_loader,
            drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
            edges_dt,
            edges_dp, w_dp,
            edges_pc, w_pc,
            edges_pt,
        )

        val_loss, m_va, aux_va = eval_split(
            model, val_loader, y_scaler,
            drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
            edges_dt,
            edges_dp, w_dp,
            edges_pc, w_pc,
            edges_pt,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        g_drug_mean = float(aux_va["g_drug"].mean().detach().cpu())

        print(
            f"\nEpoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  "
            f"val_RMSE(raw)={val_loss:.4f}  "
            f"g_drug_mean={g_drug_mean:.3f}"
        )
        print_metrics("[Val(raw)]", m_va)

        if val_loss < best_val - MIN_DELTA:
            best_val, best_epoch, patience_cnt = val_loss, epoch, 0
            torch.save(model.state_dict(), BEST_PATH)
        else:
            patience_cnt += 1

        if EARLY_STOPPING and patience_cnt >= PATIENCE:
            print(f"\n[EarlyStopping] Stop at epoch {epoch}, best epoch={best_epoch}, best val RMSE={best_val:.4f}")
            break

    print(f"\n[Best] epoch={best_epoch} best_val_rmse={best_val:.4f}")


    print("\n================= Final Test (Best Val RMSE in raw space) =================")
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))

    test_loss, m_te, aux_te = eval_split(
        model, test_loader, y_scaler,
        drug_graph, X_targ, X_cell_mut, X_cell_cnv, X_cell_rna,
        edges_dt,
        edges_dp, w_dp,
        edges_pc, w_pc,
        edges_pt,
    )
    print_metrics("[Test(raw)]", m_te)
    print(f"[Saved] {BEST_PATH}\n[Saved] {Y_SCALER_PATH}")

    if SAVE_ATTN:
        attn_pack = {
            "edges_dt_target2drug": tuple(to_np(x) for x in edges_dt["target2drug"]),
            "edges_dp_pathway2drug": tuple(to_np(x) for x in edges_dp["pathway2drug"]),
            "edges_pc_pathway2cell": tuple(to_np(x) for x in edges_pc["pathway2cell"]),
            "edges_pt_pathway2target": tuple(to_np(x) for x in edges_pt["pathway2target"]),
            "edges_pt_target2pathway": tuple(to_np(x) for x in edges_pt["target2pathway"]),
            "attn_dt": {k: to_np(v) for k, v in aux_te["attn_dt"].items()},
            "attn_dp": {k: to_np(v) for k, v in aux_te["attn_dp"].items()},
            "attn_pc": {k: to_np(v) for k, v in aux_te["attn_pc"].items()},
            "attn_pt": {k: to_np(v) for k, v in aux_te["attn_pt"].items()},
            "g_drug": to_np(aux_te["g_drug"]),
        }
        with open(ATTN_PATH, "wb") as f:
            pickle.dump(attn_pack, f)
        print(f"[Saved] attention & gate -> {ATTN_PATH}")


if __name__ == "__main__":
    main()
