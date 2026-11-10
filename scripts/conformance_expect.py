#!/usr/bin/env python3
"""Merge device conformance summaries and list the NEW failures.

A device conformance run (``pytest tests/test_conformance.py --backend nki
--conformance-device neuron``) marks known platform limits as
``xfail(strict=False)`` with a ``[device-known:<category>]`` reason, and prints
a summary at the end. Runs on a Trainium box are split into chunks, so pass
``--conformance-json PATH`` to each chunk and merge them here::

    pytest tests/test_conformance.py --backend nki --conformance-device neuron \\
        -k "<chunk>" --conformance-json chunk1.json
    python scripts/conformance_expect.py chunk*.json

Prints passed, xfailed by category, XPASS by category (a known category that
now passes: a signal, not a failure) and the NEW failures. Exits 1 if there
are NEW failures, else 0.
"""

from __future__ import annotations

import argparse
import json
import sys


def merge(paths):
    out = {"passed": 0, "skipped": 0, "xfailed": {}, "xpassed": {}, "new_failures": []}
    for p in paths:
        with open(p) as fh:
            s = json.load(fh)
        out["passed"] += s["passed"]
        out["skipped"] += s["skipped"]
        for key in ("xfailed", "xpassed"):
            for cat, ids in s[key].items():
                out[key].setdefault(cat, []).extend(ids)
        out["new_failures"].extend(s["new_failures"])
    return out


def report(s, verbose=False):
    lines = ["conformance expectations (device run)"]
    lines.append(f"  passed: {s['passed']}   skipped: {s['skipped']}")
    for key, label in (("xfailed", "xfailed"), ("xpassed", "XPASS (known category now passes)")):
        lines.append(f"  {label}: {sum(len(v) for v in s[key].values())}")
        for cat, ids in sorted(s[key].items()):
            lines.append(f"    {cat}: {len(ids)}")
            if verbose or key == "xpassed":
                lines.extend(f"      {i}" for i in ids)
    lines.append(f"  NEW failures: {len(s['new_failures'])}")
    lines.extend(f"    {i}" for i in s["new_failures"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json", nargs="+", help="--conformance-json files from device runs")
    ap.add_argument("-v", "--verbose", action="store_true", help="list xfailed IDs too")
    args = ap.parse_args(argv)
    s = merge(args.json)
    print(report(s, args.verbose))
    return 1 if s["new_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
