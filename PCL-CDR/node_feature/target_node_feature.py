from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel

IN_PATH = Path("../MultiVG-DRP/drug_targetdrug_target/uniprot_human_target_sequences_192.csv")

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

POOLING = "mean"
CACHE_DIR = r"../MultiVG-DRP/node_feature"

MAX_TOKENS_PER_BATCH = 6000
MAX_SEQS_PER_BATCH = 32

OUT_CSV = IN_PATH.with_name(f"esm2_hf_{MODEL_NAME.replace('/', '_')}_{POOLING}_embeddings.csv")

AA_ALLOWED = set(list("ACDEFGHIKLMNPQRSTVWYBXZJUO"))


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = re.sub(r"[^A-Z]", "", seq)
    if not seq:
        return ""
    seq = "".join([ch if ch in AA_ALLOWED else "X" for ch in seq])
    return seq


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


def chunk_sequence(seq: str, max_len: int):
    L = len(seq)
    if L <= max_len:
        return [(seq, L)]
    chunks = []
    start = 0
    while start < L:
        end = min(start + max_len, L)
        sub = seq[start:end]
        chunks.append((sub, len(sub)))
        start = end
    return chunks


def make_batches(items, max_tokens=MAX_TOKENS_PER_BATCH, max_seqs=MAX_SEQS_PER_BATCH):
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)

    batches = []
    cur = []
    cur_tok = 0

    for lab, seq in items:
        tok = len(seq) + 2
        if cur and (cur_tok + tok > max_tokens or len(cur) >= max_seqs):
            batches.append(cur)
            cur = []
            cur_tok = 0
        cur.append((lab, seq))
        cur_tok += tok

    if cur:
        batches.append(cur)
    return batches


def main():
    enc = open_with_fallback_encodings(IN_PATH)
    df = pd.read_csv(IN_PATH, encoding=enc)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    cols_lower = {str(c).lower().lstrip("\ufeff"): c for c in df.columns}
    if "protein" not in cols_lower or "sequence" not in cols_lower:
        raise ValueError(f"Required columns are protein and sequence. Actual columns: {list(df.columns)}")

    prot_col = cols_lower["protein"]
    seq_col = cols_lower["sequence"]

    proteins, seqs = [], []
    for _, r in df.iterrows():
        p = str(r[prot_col]).strip()
        s = clean_sequence(r[seq_col])
        if p and s:
            proteins.append(p)
            seqs.append(s)

    if not proteins:
        raise RuntimeError("No valid protein/sequence records were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    print(f"[Load] tokenizer/model: {MODEL_NAME}")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = EsmModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)

    max_pos = getattr(model.config, "max_position_embeddings", 1026)
    max_aa_per_chunk = max_pos - 2
    print(
        f"[Model] hidden_size={model.config.hidden_size}, "
        f"max_position_embeddings={max_pos}, "
        f"max_aa_per_chunk={max_aa_per_chunk}"
    )

    chunk_items = []
    for p, s in zip(proteins, seqs):
        chunks = chunk_sequence(s, max_aa_per_chunk)
        for i, (sub, L) in enumerate(chunks):
            chunk_items.append((f"{p}|chunk{i}", sub))

    print(f"[Input] proteins={len(proteins)} | chunks={len(chunk_items)}")

    batches = make_batches(chunk_items, MAX_TOKENS_PER_BATCH, MAX_SEQS_PER_BATCH)
    print(f"[Batch] num_batches={len(batches)}")

    sum_vec = {}
    sum_len = {}

    with torch.no_grad():
        for bi, batch in enumerate(batches, 1):
            labels = [x[0] for x in batch]
            seq_list = [x[1] for x in batch]

            inputs = tokenizer(
                seq_list,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            use_amp = device.type == "cuda"
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(**inputs)
            else:
                out = model(**inputs)

            hidden = out.last_hidden_state
            attn = inputs["attention_mask"]

            hidden = hidden.float().cpu()
            attn = attn.cpu()

            for i, lab in enumerate(labels):
                protein = lab.split("|chunk")[0]

                valid_len = int(attn[i].sum().item())
                L = max(valid_len - 2, 0)

                if L <= 0:
                    continue

                if POOLING == "bos":
                    vec = hidden[i, 0, :].numpy()
                    chunk_len = L
                else:
                    vec = hidden[i, 1:valid_len - 1, :].mean(dim=0).numpy()
                    chunk_len = L

                if protein not in sum_vec:
                    sum_vec[protein] = vec * chunk_len
                    sum_len[protein] = chunk_len
                else:
                    sum_vec[protein] += vec * chunk_len
                    sum_len[protein] += chunk_len

            if bi % 10 == 0 or bi == len(batches):
                print(f"  done {bi}/{len(batches)} batches")

    proteins_out, X = [], []
    for p in proteins:
        if p in sum_vec:
            vec = sum_vec[p] / max(sum_len[p], 1)
            proteins_out.append(p)
            X.append(vec)

    X = np.vstack(X)
    dim = X.shape[1]
    print(f"[Output] proteins={X.shape[0]} | dim={dim}")

    out_df = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(dim)])
    out_df.insert(0, "protein", proteins_out)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUT_CSV}")


if __name__ == "__main__":
    main()