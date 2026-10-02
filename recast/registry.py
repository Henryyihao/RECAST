from __future__ import annotations
import torch

PATHS = {
    "chronos2": "models/chronos2",
    "bolt_s": "models/chronosbolt",
    "bolt_b": "models/amazon_chronos-bolt-base",
    "toto": "models/Datadog_Toto-2.0-313m",
    "toto22": "models/Datadog_Toto-2.0-22m",
    "timemoe": "models/Maple728_TimeMoE-50M",
}

NMAX_OVERRIDE = {}

QIDX = {"chronos2": [2, 4, 6, 8, 10, 12, 14, 16, 18], "bolt_s": list(range(9)), "bolt_b": list(range(9)),
        "toto": list(range(9)), "toto22": list(range(9)), "timemoe": list(range(9))}


def build_model(arch: str, backbone=None, age_channel=True, age_bias=True, shared_norm=True, age_attention=True, query_row=None,
                edge=True, now_weight=1.0, evidence_gate=False, separate_now_head=False, forecast_anchor=False, mult_head=False, no_now_gate=False, **kw):
    path = backbone or PATHS[arch]
    m = _build(arch, path, age_channel, age_bias, shared_norm, age_attention, query_row)
    m.edge = edge
    m.no_now_gate = no_now_gate
    m.now_weight = now_weight
    m.evidence_gate = evidence_gate
    m.separate_now_head = separate_now_head
    m.forecast_anchor = forecast_anchor
    m.mult_head = mult_head
    m.old_input_cols = _old_input_cols(arch, m)
    return m


def _old_input_cols(arch, m):

    out = []
    if arch == "chronos2":
        p = m.p
        for lin in (m.m.input_patch_embedding.hidden_layer, m.m.input_patch_embedding.residual_layer):
            if lin.weight.shape[1] == 6 * p:
                out.append((lin.weight, 3 * p))
    elif arch in ("bolt_s", "bolt_b"):
        p = m.p
        for lin in (m.m.input_patch_embedding.hidden_layer, m.m.input_patch_embedding.residual_layer):
            if lin.weight.shape[1] == 5 * p:
                out.append((lin.weight, 2 * p))
    elif arch in ("toto", "toto22"):
        p = m.p
        for mod in m.m.patch_proj.modules():
            if isinstance(mod, torch.nn.Linear) and mod.weight.shape[1] == 5 * p:
                out.append((mod.weight, 2 * p))
    elif arch == "timemoe":
        emb = m.m.model.embed_layer
        for lin in (emb.emb_layer, emb.gate_layer):
            if lin.weight.shape[1] > 1:
                out.append((lin.weight, 1))
    return out


def _build(arch, path, age_channel, age_bias, shared_norm, age_attention, query_row):
    if arch == "chronos2":
        from .models.va_chronos2 import VAChronos2
        m = VAChronos2(path, age_channel=age_channel, age_bias=age_bias, shared_norm=shared_norm)
        m.input_embedding_parameters = lambda: list(m.m.input_patch_embedding.parameters())
        m.output_head_parameters = lambda: list(m.m.output_patch_embedding.parameters())
        m.target_row_default = 1 if query_row is None else query_row
        return m
    if arch in ("bolt_s", "bolt_b"):
        from .models.va_bolt import VABolt
        m = VABolt(path, age_channel=age_channel, age_bias=age_bias, shared_norm=shared_norm, age_attention=age_attention,
                   query_row=1 if query_row is None else query_row)
        m.input_embedding_parameters = lambda: list(m.m.input_patch_embedding.parameters())
        m.output_head_parameters = lambda: list(m.m.output_patch_embedding.parameters())
        m.target_row_default = 1 if query_row is None else query_row
        return m
    if arch in ("toto", "toto22"):
        from .models.va_toto import VAToto
        m = VAToto(path, age_channel=age_channel, age_bias=age_bias, shared_norm=shared_norm, age_attention=age_attention,
                   query_row=1 if query_row is None else query_row)
        return m
    if arch == "timemoe":
        from .models.va_timemoe import VATimeMoE
        m = VATimeMoE(path, age_channel=age_channel, age_bias=age_bias, shared_norm=shared_norm, age_attention=age_attention,
                      query_row=1 if query_row is None else query_row)
        return m
    raise ValueError(arch)
