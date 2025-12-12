import argparse
import json
import math
from typing import Any, Dict, List, Optional


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    return [r for r in rows if isinstance(r, dict) and r.get("event") != "warm_retriever_init"]


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _summ(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "max": max(values),
    }


def _fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0f}ms"


def _fmt_pct_speedup(after: Optional[float], before: Optional[float]) -> str:
    """
    Percent speed-up where LOWER is better (latency, call count).
    Formula: (before/after - 1) * 100
    Positive means faster / fewer; negative means regression.
    """
    if after is None or before is None or after == 0:
        return "n/a"
    pct = (before / after - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _extract(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _success_rate(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "n/a"
    ok = sum(1 for r in rows if r.get("success") is True)
    return f"{ok}/{len(rows)}"


def main() -> int:
    p = argparse.ArgumentParser(description="Compare before/after perf JSON files and print a markdown report.")
    p.add_argument("--before", required=True, help="Path to baseline perf JSON.")
    p.add_argument("--after", required=True, help="Path to optimized perf JSON.")
    p.add_argument("--out", default="", help="Optional output markdown path.")
    args = p.parse_args()

    before = _load_rows(args.before)
    after = _load_rows(args.after)

    before_total = _summ(_extract(before, "total_ms"))
    after_total = _summ(_extract(after, "total_ms"))

    before_llm = _summ(_extract(before, "llm_ms"))
    after_llm = _summ(_extract(after, "llm_ms"))

    before_neo4j = _summ(_extract(before, "neo4j_ms"))
    after_neo4j = _summ(_extract(after, "neo4j_ms"))

    before_calls = _summ(_extract(before, "llm_calls"))
    after_calls = _summ(_extract(after, "llm_calls"))

    lines = []
    lines.append("## Global retriever performance: before vs after")
    lines.append("")
    lines.append(f"- **Before**: `{args.before}` (success { _success_rate(before) })")
    lines.append(f"- **After**: `{args.after}` (success { _success_rate(after) })")
    lines.append("")

    lines.append("### Summary (p50 / p95)")
    lines.append("")
    lines.append("| Metric | Before p50 | After p50 | Speed-up | Before p95 | After p95 | Speed-up |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| total_ms | {_fmt_ms(before_total['p50'])} | {_fmt_ms(after_total['p50'])} | {_fmt_pct_speedup(after_total['p50'], before_total['p50'])} | "
        f"{_fmt_ms(before_total['p95'])} | {_fmt_ms(after_total['p95'])} | {_fmt_pct_speedup(after_total['p95'], before_total['p95'])} |"
    )
    lines.append(
        f"| llm_ms | {_fmt_ms(before_llm['p50'])} | {_fmt_ms(after_llm['p50'])} | {_fmt_pct_speedup(after_llm['p50'], before_llm['p50'])} | "
        f"{_fmt_ms(before_llm['p95'])} | {_fmt_ms(after_llm['p95'])} | {_fmt_pct_speedup(after_llm['p95'], before_llm['p95'])} |"
    )
    lines.append(
        f"| neo4j_ms | {_fmt_ms(before_neo4j['p50'])} | {_fmt_ms(after_neo4j['p50'])} | {_fmt_pct_speedup(after_neo4j['p50'], before_neo4j['p50'])} | "
        f"{_fmt_ms(before_neo4j['p95'])} | {_fmt_ms(after_neo4j['p95'])} | {_fmt_pct_speedup(after_neo4j['p95'], before_neo4j['p95'])} |"
    )
    lines.append(
        f"| llm_calls | {before_calls['p50'] if before_calls['p50'] is not None else 'n/a'} | {after_calls['p50'] if after_calls['p50'] is not None else 'n/a'} | {_fmt_pct_speedup(after_calls['p50'], before_calls['p50'])} | "
        f"{before_calls['p95'] if before_calls['p95'] is not None else 'n/a'} | {after_calls['p95'] if after_calls['p95'] is not None else 'n/a'} | {_fmt_pct_speedup(after_calls['p95'], before_calls['p95'])} |"
    )
    lines.append("")

    lines.append("### Notes")
    lines.append("")
    lines.append("- `llm_ms` is wall time waiting for the provider response (network + queueing + compute + generation).")
    lines.append("- `neo4j_ms` is time to fetch the selected communities; if `null` in baseline runs, that implementation didn’t surface timing breakdown.")
    lines.append("")

    md = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote: {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


