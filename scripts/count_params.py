import sys, torch
sys.path.insert(0, ".")
from recast.registry import build_model
for arch in ["chronos2", "bolt_s", "bolt_b", "toto", "timemoe"]:
    m = build_model(arch, age_channel=True, age_bias=True, shared_norm=True, age_attention=True, query_row=1, edge=True, evidence_gate=True, separate_now_head=True, forecast_anchor=True, mult_head=True)
    total = sum(p.numel() for p in m.parameters())
    new = sum(p.numel() for p in m.new_parameters())
    newcols = sum(w.shape[0] * (w.shape[1] - n_old) for w, n_old in m.old_input_cols)
    names = {}
    for p in m.new_parameters():
        for n, q in m.named_parameters():
            if q is p:
                key = n.split(".")[0] if not n.startswith("model") else ".".join(n.split(".")[:4]); names[key] = names.get(key, 0) + p.numel(); break
    print(f"{arch}: total {total/1e6:.2f}M  new_params {new/1e6:.2f}M  new input columns {newcols/1e6:.3f}M  => new total {(new+newcols)/1e6:.2f}M ({100*(new+newcols)/(total-new-newcols):.1f}% of backbone)")
    top = sorted(names.items(), key=lambda kv: -kv[1])[:6]; print("   ", {k: round(v/1e6, 2) for k, v in top})
