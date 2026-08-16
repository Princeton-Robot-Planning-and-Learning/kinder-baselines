"""Cross-tabulate outcome against which sampling attempt the toss was accepted on."""

import glob
import json
import sys
from collections import Counter

rows = []
for p in sorted(glob.glob(sys.argv[1])):
    for line in open(p):
        rows.append(json.loads(line))
rows.sort(key=lambda r: r["seed"])

TOSS = "move_to_toss_location_and_toss"
PICK = "pick_cube"

print(
    f"{'seed':>5} {'outcome':>20} {'pick_att':>9} {'toss_att':>9} "
    f"{'toss_acc_on':>12} {'final_cube_x':>13} {'final_cube_z':>13}"
)
tab = Counter()
for r in rows:
    ap = r.get("attempts_per_op") or {}
    ac = r.get("accepted_attempt_per_op") or {}
    fc = r.get("final_cube") or [float("nan")] * 3
    acc = ac.get(TOSS)
    print(
        f"{r['seed']:>5} {r['outcome']:>20} {ap.get(PICK, '-'):>9} {ap.get(TOSS, '-'):>9} "
        f"{str(acc):>12} {fc[0]:13.4f} {fc[2]:13.4f}"
    )
    if r["outcome"] in ("planned_and_scored", "planned_not_scored"):
        key = ("accepted_on_1" if acc == 1 else f"accepted_on_{acc}", r["outcome"])
        tab[key] += 1

print("\n=== contingency: which attempt the toss was accepted on vs outcome")
n_first_scored = sum(v for k, v in tab.items() if k[0] == "accepted_on_1" and k[1] == "planned_and_scored")
n_first_not = sum(v for k, v in tab.items() if k[0] == "accepted_on_1" and k[1] == "planned_not_scored")
n_later_scored = sum(v for k, v in tab.items() if k[0] != "accepted_on_1" and k[1] == "planned_and_scored")
n_later_not = sum(v for k, v in tab.items() if k[0] != "accepted_on_1" and k[1] == "planned_not_scored")
tot = n_first_scored + n_first_not + n_later_scored + n_later_not
print(f"  toss accepted on attempt 1 : scored {n_first_scored}/{n_first_scored + n_first_not}")
print(f"  toss accepted on attempt >=2: scored {n_later_scored}/{n_later_scored + n_later_not}")
print(f"  total refined seeds: {tot}")
print("\nraw:", dict(tab))
print("\noutcomes:", Counter(r["outcome"] for r in rows), f"n={len(rows)}")
