from pathlib import Path
import math
import pandas as pd

DRUG_TARGET_PATH = Path(r"...MultiVG-DRP/drug_target/drug_target_pubchem_clean_241.csv")
PATHWAY_PATH     = Path(r"../PRISM/raw/pathways_matched_cnv_1966.csv")

OUT_DP = PATHWAY_PATH.with_name("PRISM_drug_pathway_edges_topN.csv")

TOP_N = 30
MIN_OVERLAP = 1
USE_IDF = True


def open_with_fallback_encodings(path, encodings=("utf-8-sig", "utf-8", "gbk", "cp936", "latin1")):
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(2048)
            return enc
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def norm_gene(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = " ".join(s.split())
    return s.upper()


def load_drug_targets(path: Path) -> dict[str, list[str]]:
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    if "drug_name" not in cols_lower:
        raise ValueError(f"Cannot find the 'drug_name' column in the drug target table: {list(df.columns)[:20]}")
    drug_col = cols_lower["drug_name"]

    drug2targets = {}
    for _, row in df.iterrows():
        drug = str(row[drug_col]).strip()
        if not drug:
            continue

        tgts = []
        for c, v in row.items():
            if c == drug_col:
                continue
            if pd.isna(v):
                continue
            g = norm_gene(v)
            if not g or g.lower() in {"nan", "none"}:
                continue
            if g not in tgts:
                tgts.append(g)

        drug2targets[drug] = tgts

    return drug2targets


def load_pathways(path: Path) -> list[tuple[str, list[str]]]:
    enc = open_with_fallback_encodings(path)
    df = pd.read_csv(path, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    if "genes" not in cols_lower or "pathway" not in cols_lower:
        raise ValueError(f"The pathway table must contain 'genes' and 'pathway' columns: {list(df.columns)}")

    genes_col = cols_lower["genes"]
    pathway_col = cols_lower["pathway"]

    pathways = []
    for _, row in df.iterrows():
        pname = str(row[pathway_col]).strip()
        genes_raw = row[genes_col]
        if pd.isna(genes_raw):
            genes = []
        else:
            genes = [norm_gene(x) for x in str(genes_raw).split("|")]
            genes = [g for g in genes if g and g.lower() not in {"nan", "none"}]
            seen = set()
            uniq = []
            for g in genes:
                if g not in seen:
                    seen.add(g)
                    uniq.append(g)
            genes = uniq

        pathways.append((pname, genes))

    return pathways


def build_idf(pathways: list[tuple[str, list[str]]]) -> dict[str, float]:
    df_count = {}
    for _, genes in pathways:
        for g in set(genes):
            df_count[g] = df_count.get(g, 0) + 1

    P = len(pathways)
    idf = {g: math.log((P + 1) / (df + 1)) + 1.0 for g, df in df_count.items()}
    return idf


def main():
    drug2targets = load_drug_targets(DRUG_TARGET_PATH)
    pathways = load_pathways(PATHWAY_PATH)

    print(f"Drugs: {len(drug2targets)}")
    print(f"Pathways: {len(pathways)}")

    idf = build_idf(pathways) if USE_IDF else None

    edges = []
    for drug, targets in drug2targets.items():
        T = set(targets)
        if not T:
            continue

        denom = len(targets)

        for pname, genes in pathways:
            G = set(genes)
            inter = T & G
            k = len(inter)
            if k < MIN_OVERLAP:
                continue

            if USE_IDF:
                numer = sum(idf.get(t, 1.0) for t in inter)
            else:
                numer = float(k)

            w_dp = numer / float(denom)

            edges.append((drug, pname, w_dp, k))

    df_edges = pd.DataFrame(edges, columns=["drug_name", "pathway", "w_dp", "overlap_count"])
    if df_edges.empty:
        print("No drug-pathway edges were found. Please check MIN_OVERLAP or whether gene names are consistent.")
        return

    df_top = (
        df_edges.sort_values(["drug_name", "w_dp"], ascending=[True, False])
                .groupby("drug_name", as_index=False)
                .head(TOP_N)
                .reset_index(drop=True)
    )

    df_top.to_csv(OUT_DP, index=False, encoding="utf-8-sig")

    print(f"Original number of D-P edges: {len(df_edges)}")
    print(f"Number of D-P edges after Top-{TOP_N} filtering: {len(df_top)}")
    print(f"Output file: {OUT_DP}")


if __name__ == "__main__":
    main()