"""Diff two checkpoints tensor by tensor; for the embedding, report changed rows with scale factors."""
import json
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

a_path, b_path, tok_dir = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
out = {"tensors": {}, "embedding_rows": []}
with safe_open(a_path, "pt") as fa, safe_open(b_path, "pt") as fb:
    for k in fa.keys():
        a, b = fa.get_tensor(k), fb.get_tensor(k)
        n = int((a != b).sum())
        if n:
            out["tensors"][k] = {"changed": n, "numel": a.numel()}
            if k.endswith("embed_tokens.weight"):
                rows = torch.nonzero((a != b).any(1)).flatten().tolist()
                for r in rows:
                    ra, rb = a[r].float(), b[r].float()
                    ratio = (rb / ra.where(ra.abs() > 1e-3, torch.ones_like(ra)))
                    out["embedding_rows"].append({"id": r, "token": tok.decode([r]), "repr": repr(tok.convert_ids_to_tokens(r)),
                                                  "norm_ratio": round((rb.norm() / ra.norm()).item(), 4),
                                                  "cos": round((ra @ rb / (ra.norm() * rb.norm())).item(), 5),
                                                  "ratio_median": round(ratio.median().item(), 4)})
            elif a.ndim == 1:
                ra, rb = a.float(), b.float()
                out["tensors"][k]["ratio_of_1plus_w_median"] = round(((1 + rb) / (1 + ra)).median().item(), 4)
                out["tensors"][k]["ratio_of_1plus_w_std"] = round(((1 + rb) / (1 + ra)).std().item(), 5)
print(json.dumps(out, indent=1, ensure_ascii=False))
