"""Standalone autograder — no platform required.
Runs each assignment's pytest suites against the student's code and prints a
score per assignment. Each assignment's points are split into three buckets
(r2.1 grading scheme):

    public   60%   tests/                    ships to students; the spec
    holdout  20%   tests_private/holdout/    same behaviours, different inputs
    hard     20%   tests_private/hard/       harder cases; every hard category
                                             has a simpler public counterpart
                                             (tests/test_reflect_*)

Within a bucket, points = fraction of tests passed × bucket points. The
tests_private/ directory never ships (scripts/export_student_repo.py refuses
to), so in a student checkout only the public bucket exists: grade.py then
reports the public 60% and says the remaining 40% is graded by the instructor.

Usage:
    python grade.py                     # grade all assignments (starter/)
    python grade.py a4 a5               # grade specific assignments
    python grade.py --solutions         # sanity-check reference solutions
    python grade.py --public-only       # what a student sees
    python grade.py a5 --json out.json  # machine-readable results
Students work directly in each assignment's starter/ directory. Instructors
running against their own reference code use --solutions.

Assignments are looked up next to this file: `assignments/current/` at the
instructor repo's root, `assignments/` inside `release/` and in the student repo.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(here=HERE):
    """Where the assignments live relative to this file: `assignments/current/`
    at the instructor repo's root, plain `assignments/` inside `release/` or in
    the exported student repo (the tools all share this rule)."""
    for cand in (os.path.join(here, "assignments", "current"),
                 os.path.join(here, "assignments")):
        if os.path.isdir(cand):
            return cand
    sys.exit(f"no assignments/ directory next to {here}")


ROOT = resolve()

WEIGHTS = {  # course points per assignment (autograded portion)
    "a0-setup": 0,
    "a1-linear-regression": 8,
    "a2-classification": 8,
    "a3-autodiff": 12,
    "a4-bpe-tokenizer": 10,
    "a5-transformer": 20,
    "a6-training": 10,
    "a7-data-curation": 10,   # EXTRA CREDIT in the 15-week plan (not in the 130-pt core)
    "a8-sft": 10,             # EXTRA CREDIT as of r2 (not in the 130-pt core)
}


BUCKETS = [  # (label, test directory relative to the assignment, share of points)
    ("public", "tests", 0.60),
    ("holdout", os.path.join("tests_private", "holdout"), 0.20),
    ("hard", os.path.join("tests_private", "hard"), 0.20),
]


def run_suite(path, test_dir, submission_dir):
    """Run one pytest directory; return (passed, total, output_tail)."""
    env = dict(os.environ, SUBMISSION_DIR=submission_dir)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-q", "--tb=line", "-p",
         "no:cacheprovider"],
        cwd=path, env=env, capture_output=True, text=True, timeout=1200)
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", out)
    errors = int(m.group(1)) if m else 0
    return passed, passed + failed + errors, out.strip().splitlines()[-12:]


def run_assignment(name, submission_dir, public_only=False):
    path = os.path.join(ROOT, name)
    weight = WEIGHTS[name]
    buckets, score, tail, private_seen = {}, 0.0, [], False
    for label, test_dir, share in BUCKETS:
        if not os.path.isdir(os.path.join(path, test_dir)):
            buckets[label] = None          # absent (student checkout, or a0)
            continue
        if public_only and label != "public":
            buckets[label] = None
            continue
        passed, total, t = run_suite(path, test_dir, submission_dir)
        pts = round(weight * share * passed / total, 2) if total else 0.0
        buckets[label] = {"passed": passed, "total": total,
                          "points": pts, "max": round(weight * share, 2)}
        score += pts
        if passed != total:
            tail += [f"[{label}]"] + t
        if label != "public":
            private_seen = True
    return {"assignment": name, "buckets": buckets, "score": round(score, 2),
            "max": weight, "private_graded": private_seen,
            "output_tail": tail}


def fmt_bucket(b):
    return "  --  " if b is None else f"{b['passed']:>3}/{b['total']:<3}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assignments", nargs="*",
                    help="short names like a4, or full dir names")
    ap.add_argument("--solutions", action="store_true",
                    help="grade reference solutions instead of starter code")
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--public-only", action="store_true",
                    help="run only tests/ (what a student checkout can run)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    all_names = sorted(d for d in os.listdir(ROOT)
                       if os.path.isdir(os.path.join(ROOT, d)))
    if args.assignments:
        chosen = []
        for a in args.assignments:
            matches = [n for n in all_names if n == a or n.startswith(a + "-")
                       or n.startswith(a)]
            if not matches:
                sys.exit(f"unknown assignment: {a}")
            chosen.append(matches[0])
    else:
        chosen = all_names

    sub = "solution" if args.solutions else "starter"
    results, got, mx = [], 0.0, 0
    print(f"grading against: {sub}/   (buckets: public 60% | holdout 20% | hard 20%)\n")
    print(f"    {'assignment':24s} {'public':>8} {'holdout':>8} {'hard':>8}   points")
    any_private = False
    for name in chosen:
        r = run_assignment(name, sub, public_only=args.public_only)
        results.append(r)
        got += r["score"]
        mx += r["max"]
        any_private = any_private or r["private_graded"]
        b = r["buckets"]
        all_ok = all(x is None or x["passed"] == x["total"] for x in b.values()) \
            and any(x is not None and x["total"] for x in b.values())
        status = "OK " if all_ok else "    "
        print(f"{status}{name:24s} {fmt_bucket(b['public']):>8} "
              f"{fmt_bucket(b['holdout']):>8} {fmt_bucket(b['hard']):>8}"
              f"   {r['score']:>6.2f}/{r['max']} pts")
        if args.verbose and r["output_tail"]:
            print("      " + "\n      ".join(r["output_tail"]))
    print(f"\nTOTAL: {got:.2f}/{mx} autograded points")
    if not any_private:
        print("(public bucket only: the remaining 40% -- holdout 20% + hard 20% -- "
              "is graded by the instructor on private tests of the same kinds)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results, "total": got, "max": mx}, f,
                      indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
