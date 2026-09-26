"""python -m src.profile [--execute] [--limit N] [--until STAGE] | --verify | --spot-check N

Dry-run is the default: prints token and USD estimates, makes no paid calls."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import orjson

from src.profile import (CATEGORIES, LEDGER_PATH, MODEL, NORMALIZED, PROFILE_DIR,
                         REDUCED_PATH, SELECTION_PATH, SPEND_PATH)
from src.profile import extract as ex
from src.profile import reduce as rd
from src.profile import render as rn
from src.profile.openrouter import is_cached, model_prices
from src.profile.seed import seed_lines
from src.profile.select import selection

STAGES = ("extract", "reduce", "render")
# Output/input ratios for stages whose inputs don't exist until the previous stage runs.
EXTRACT_OUT_PER_TURN = 20      # pilot: ~4 tok/turn on long turns, >16 tok/turn on short ones
TOKENIZER_FUDGE = 1.3          # pilot: provider counted 1.3x chars/4
REDUCE_OUT_RATIO = 0.2
RENDER_CLAIMS_RATIO = 0.6       # share of extract output surviving dedupe + confidence floor
RENDER_OUT_TOKENS = 13_000      # core + PROFILE.md + 3 domains + 2 overlays
REASONING_OVERHEAD = 1.5        # render runs with reasoning effort "low"


def _tok(obj) -> int:
    return len(orjson.dumps(obj)) // 4


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")


def _read_jsonl(path) -> list[dict]:
    with path.open("rb") as f:
        return [orjson.loads(l) for l in f]


def _print_counts(title: str, counts: Counter) -> None:
    print(f"{title}:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


def estimate(batches, seeds) -> None:
    prices = model_prices(MODEL)
    cached = sum(is_cached(ex.payload(b)) for b in batches)
    ex_in = int(sum(_tok(ex.payload(b)) for b in batches) * TOKENIZER_FUDGE)
    n_turns = sum(len(c["turns"]) for b in batches for c in b["conversations"])
    ex_out = n_turns * EXTRACT_OUT_PER_TURN
    rd_in = int(ex_out * 1.1) + 300 * max(1, len(CATEGORIES) * 2)
    rd_out = int(rd_in * REDUCE_OUT_RATIO * REASONING_OVERHEAD)
    seed_tok = {name: _tok(rn.render_inputs([], seeds, spec["accounts"], spec["categories"]))
                for name, spec in rn.SPECS.items()}
    rn_in = int(sum(seed_tok.values()) * TOKENIZER_FUDGE) + int(ex_out * RENDER_CLAIMS_RATIO) * len(rn.SPECS) + 400 * len(rn.SPECS)
    rn_out = int(RENDER_OUT_TOKENS * REASONING_OVERHEAD)
    rows = [("extract", ex_in, ex_out), ("reduce", rd_in, rd_out), ("render", rn_in, rn_out)]
    lo = min(prices, key=lambda p: p[0] + p[1])
    hi = max(prices, key=lambda p: p[0] + p[1])
    print(f"\nDRY RUN, model {MODEL}, {len(prices)} endpoints, "
          f"${lo[0] * 1e6:.3f}-{hi[0] * 1e6:.3f}/M in, ${lo[1] * 1e6:.3f}-{hi[1] * 1e6:.3f}/M out")
    print(f"  extract batches: {len(batches)} ({cached} already cached)")
    tot_lo = tot_hi = 0.0
    for stage, i, o in rows:
        a, b = i * lo[0] + o * lo[1], i * hi[0] + o * hi[1]
        tot_lo, tot_hi = tot_lo + a, tot_hi + b
        print(f"  {stage:8} in {i:>10,} tok  out {o:>9,} tok  ${a:.4f} - ${b:.4f}")
    print(f"  {'total':8} {'':>34}${tot_lo:.4f} - ${tot_hi:.4f}")
    print(f"  extract: {n_turns:,} user turns. Input is chars/4 x {TOKENIZER_FUDGE}; the rest is "
          "estimated from ratios in __main__.py")


def verify() -> int:
    """Every ledger line validates; every extract conv_uid/msg_i resolves to a user turn."""
    ledger = _read_jsonl(LEDGER_PATH)
    user_turns: dict[str, set[int]] = {}
    with (NORMALIZED / "conversations.jsonl").open("rb") as f:
        for raw in f:
            conv = orjson.loads(raw)
            user_turns[conv["uid"]] = {m["i"] for m in conv["messages"] if m["role"] == "user"}
    required = {"id", "claim", "category", "confidence", "account", "conv_uid", "msg_i", "ts", "source", "model"}
    bad: Counter[str] = Counter()
    ids = Counter(l.get("id") for l in ledger)
    for l in ledger:
        if missing := required - l.keys():
            bad[f"missing:{','.join(sorted(missing))}"] += 1
            continue
        if ids[l["id"]] > 1:
            bad["duplicate_id"] += 1
        if not 0 <= l["confidence"] <= 1:
            bad["confidence_range"] += 1
        if l["source"] == "extract":
            if l["category"] not in CATEGORIES:
                bad["category"] += 1
            if l["conv_uid"] not in user_turns:
                bad["conv_uid_unresolved"] += 1
            elif not l["msg_i"] or any(i not in user_turns[l["conv_uid"]] for i in l["msg_i"]):
                bad["msg_i_unresolved"] += 1
        elif l["source"] != "seed":
            bad["source"] += 1
    by = Counter((l["source"], l["account"]) for l in ledger)
    print(f"ledger lines: {len(ledger)}")
    for (src, acct), n in sorted(by.items()):
        print(f"  {src:8} {acct:16} {n}")
    _print_counts("invalid", bad) if bad else print("invalid: 0")
    if SPEND_PATH.exists():
        spend = _read_jsonl(SPEND_PATH)
        print(f"spend: {len(spend)} calls, ${sum(s['cost'] or 0 for s in spend):.4f}")
    return 1 if bad else 0


def spot_check(n: int, seed: int) -> None:
    lines = [l for l in _read_jsonl(LEDGER_PATH) if l["source"] == "extract"]
    picks = random.Random(seed).sample(lines, min(n, len(lines)))
    want = {l["conv_uid"] for l in picks}
    convs = {}
    with (NORMALIZED / "conversations.jsonl").open("rb") as f:
        for raw in f:
            conv = orjson.loads(raw)
            if conv["uid"] in want:
                convs[conv["uid"]] = conv
    for l in picks:
        print(f"\n[{l['category']} {l['confidence']}] {l['claim'][:300]}")
        print(f"  {l['conv_uid']} msg_i={l['msg_i']}")
        msgs = {m["i"]: m for m in convs[l["conv_uid"]]["messages"]}
        for i in l["msg_i"]:
            print(f"  > {msgs[i]['text'][:500]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.profile")
    ap.add_argument("--execute", action="store_true", help="make paid API calls")
    ap.add_argument("--limit", type=int, help="top-N conversations by selection score")
    ap.add_argument("--until", choices=STAGES, default="render")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--spot-check", type=int, metavar="N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.verify:
        return verify()
    if args.spot_check:
        spot_check(args.spot_check, args.seed)
        return 0

    PROFILE_DIR.mkdir(exist_ok=True)
    seeds, seed_counts = seed_lines()
    _print_counts("seed", seed_counts)
    sel = selection()
    _write_jsonl(SELECTION_PATH, sel)
    print(f"select: {len(sel)} scored, {sum(r['score'] > 0 for r in sel)} with user turns")
    order = [r["uid"] for r in sel if r["score"] > 0]
    batches = ex.build_batches(order, args.limit)

    if not args.execute:
        estimate(batches, seeds)
        print("\nNo paid calls made. Re-run with --execute after approval.")
        return 0

    lines, ex_counts = ex.extract(batches)
    _write_jsonl(LEDGER_PATH, seeds + lines)
    _print_counts("extract", ex_counts)
    if verify():
        return 1
    if args.until == "extract":
        return 0
    reduced, rd_counts = rd.reduce(seeds + lines)
    _write_jsonl(REDUCED_PATH, reduced)
    _print_counts("reduce", rd_counts)
    if args.until == "reduce":
        return 0
    _print_counts("render", rn.render(reduced, seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
