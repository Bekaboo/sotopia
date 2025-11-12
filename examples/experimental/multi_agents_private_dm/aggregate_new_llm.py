"""
Aggregate LLM-judged metrics (DA, IA, Eff, CPV, Composite) across scenarios.

Looks under: <out_root>/scenario_*/metrics/{da_llm.json, ia_llm.json, eff_llm.json, cpv_llm.json, composite_llm.json}
Produces: <out_root>/aggregate/aggregated_metrics_llm.json and aggregated_metrics_llm.txt

Aggregation rules
- DA/IA/Eff/Composite: unweighted mean across scenarios where present for an agent
- CPV: pooled leaks/handled (preferred), and also mean of per-scenario cpv for reference
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Tuple


def _load(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _get_agents(d: Dict[str, Any] | None) -> set[str]:
    if not d:
        return set()
    a = d.get("agents")
    if isinstance(a, dict):
        return set(a.keys())
    return set()


def aggregate_new_llm_metrics(out_root: str) -> None:
    scenario_metrics = sorted(glob.glob(os.path.join(out_root, "scenario_*", "metrics")))
    if not scenario_metrics:
        return

    # Per-agent accumulators
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    cpv_pool: Dict[str, Tuple[int, int]] = {}  # agent -> (leaks_sum, handled_sum)

    for mdir in scenario_metrics:
        da = _load(os.path.join(mdir, "da_llm.json"))
        ia = _load(os.path.join(mdir, "ia_llm.json"))
        eff = _load(os.path.join(mdir, "eff_llm.json"))
        cpv = _load(os.path.join(mdir, "cpv_llm.json"))
        comp = _load(os.path.join(mdir, "composite_llm.json"))

        agents = _get_agents(da) | _get_agents(ia) | _get_agents(eff) | _get_agents(cpv) | _get_agents(comp)

        for agent in agents:
            # Helper to add a value
            def add(metric: str, value: Any) -> None:
                try:
                    v = float(value)
                except Exception:
                    return
                sums.setdefault(agent, {}).setdefault(metric, 0.0)
                counts.setdefault(agent, {}).setdefault(metric, 0)
                sums[agent][metric] += v
                counts[agent][metric] += 1

            # DA
            if da and isinstance(da.get("agents"), dict):
                val = da["agents"].get(agent, {}).get("mean_da")
                add("DA", val)
            # IA
            if ia and isinstance(ia.get("agents"), dict):
                val = ia["agents"].get(agent, {}).get("mean_ia")
                add("IA", val)
            # Eff
            if eff and isinstance(eff.get("agents"), dict):
                val = eff["agents"].get(agent, {}).get("efficiency")
                add("Eff", val)
            # Composite
            if comp and isinstance(comp.get("agents"), dict):
                val = comp["agents"].get(agent, {}).get("Composite")
                add("Composite", val)
            # CPV pooled and mean
            if cpv and isinstance(cpv.get("agents"), dict):
                entry = cpv["agents"].get(agent, {})
                leaks = entry.get("leaks")
                handled = entry.get("handled")
                try:
                    l = int(leaks or 0)
                    h = int(handled or 0)
                    prev_l, prev_h = cpv_pool.get(agent, (0, 0))
                    cpv_pool[agent] = (prev_l + l, prev_h + h)
                except Exception:
                    pass
                # Also track mean of per-scenario cpv
                add("CPV_mean", entry.get("cpv"))

    # Build output
    out: Dict[str, Any] = {"agents": {}, "notes": "CPV_pooled computed as total_leaks/total_handled across scenarios; CPV_mean is simple mean of per-scenario cpv"}
    lines: list[str] = []
    lines.append("Aggregated LLM Metrics across scenarios\n")
    lines.append("(Averages where applicable; CPV both pooled and mean)\n")

    all_agents = set(sums.keys()) | set(cpv_pool.keys())
    for agent in sorted(all_agents):
        def mean_of(metric: str) -> float | None:
            if agent in sums and metric in sums[agent]:
                c = counts.get(agent, {}).get(metric, 0)
                if c > 0:
                    return sums[agent][metric] / c
            return None

        DA = mean_of("DA")
        IA = mean_of("IA")
        Eff = mean_of("Eff")
        Composite = mean_of("Composite")
        CPV_mean = mean_of("CPV_mean")
        l, h = cpv_pool.get(agent, (0, 0))
        CPV_pooled = (l / h) if h > 0 else 0.0

        out["agents"][agent] = {
            "DA_mean": DA,
            "IA_mean": IA,
            "Eff_mean": Eff,
            "Composite_mean": Composite,
            "CPV_pooled": CPV_pooled,
            "CPV_mean": CPV_mean,
            "leaks_total": l,
            "handled_total": h,
        }
        lines.append(
            f"- {agent}: DA={DA if DA is not None else 'NA'}, IA={IA if IA is not None else 'NA'}, Eff={Eff if Eff is not None else 'NA'}, CPV_pooled={CPV_pooled:.3f} (leaks={l}, handled={h}), CPV_mean={CPV_mean if CPV_mean is not None else 'NA'}, Composite={Composite if Composite is not None else 'NA'}"
        )

    # Overall averages (macro-averages across agents)
    def overall_avg(key: str) -> float | None:
        vals = [v.get(key) for v in out["agents"].values() if v.get(key) is not None]
        try:
            return sum(vals) / len(vals) if vals else None
        except Exception:
            return None

    out["overall"] = {
        "DA_mean": overall_avg("DA_mean"),
        "IA_mean": overall_avg("IA_mean"),
        "Eff_mean": overall_avg("Eff_mean"),
        "Composite_mean": overall_avg("Composite_mean"),
        "CPV_pooled": overall_avg("CPV_pooled"),
        "CPV_mean": overall_avg("CPV_mean"),
    }
    lines.append("")
    lines.append("Overall (macro-averages across agents):")
    lines.append(
        f"DA={out['overall']['DA_mean'] if out['overall']['DA_mean'] is not None else 'NA'}, IA={out['overall']['IA_mean'] if out['overall']['IA_mean'] is not None else 'NA'}, Eff={out['overall']['Eff_mean'] if out['overall']['Eff_mean'] is not None else 'NA'}, CPV_pooled={out['overall']['CPV_pooled'] if out['overall']['CPV_pooled'] is not None else 'NA'}, CPV_mean={out['overall']['CPV_mean'] if out['overall']['CPV_mean'] is not None else 'NA'}, Composite={out['overall']['Composite_mean'] if out['overall']['Composite_mean'] is not None else 'NA'}"
    )

    # Persist
    agg_dir = os.path.join(out_root, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    with open(os.path.join(agg_dir, "aggregated_metrics_llm.json"), "w") as jf:
        json.dump(out, jf, ensure_ascii=False, indent=2)
    with open(os.path.join(agg_dir, "aggregated_metrics_llm.txt"), "w") as tf:
        tf.write("\n".join(lines) + "\n")

