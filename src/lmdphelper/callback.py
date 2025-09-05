# --- callbacks.py (inline or above your optimize call) ---
from pathlib import Path
import json
import pandas as pd


def leaderboard_callback(study, trial, out_dir="hpo_results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Update leaderboard.csv (top to bottom)
    # try:
    df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
    df.sort_values("value", ascending=False).to_csv(Path(out_dir)/"leaderboard.csv", index=False)

    # 2) Refresh best_config files (if we have a best yet)
    try:
        best = study.best_trial   # raises if none complete
    except ValueError:
        return

    best_dir = Path(out_dir)
    # Human readable
    lines = [
        f"Best value: {best.value}",
        f"Best trial #: {best.number}",
        "",
        "[Params]"
    ] + [f"  {k}: {v}" for k, v in sorted(best.params.items())]

    if best.user_attrs:
        lines += ["", "[User attrs]"] + [f"  {k}: {v}" for k, v in sorted(best.user_attrs.items())]

    (best_dir/"best_config.txt").write_text("\n".join(lines))

    # Machine readable (include params + attrs)
    json.dump(
        {
            "value": best.value,
            "number": best.number,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
        open(best_dir/"best_config.json","w"),
        indent=2
    )
