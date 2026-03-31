"""
Training Pipeline
=================
Runs Stage 1 → Stage 2+3 iteratively until val AUC-ROC ≥ 0.7.

Usage:
    python train_pipeline.py

On each Stage 2 round after the first, the model resumes from the best
Stage 2 checkpoint with a halved learning rate (fine-tuning continuation).
"""

import json
from pathlib import Path

from stage1_pretrain import train_stage1, CFG as STAGE1_CFG
from stage2_finetune import train_stage2, CFG as STAGE2_CFG

TARGET_AUC = 0.7
MAX_ROUNDS = 10


def main():
    # ── Stage 1 ─────────────────────────────────────────────────────────────
    stage1_ckpt = Path(STAGE1_CFG["stage1_ckpt"])
    if stage1_ckpt.exists():
        print(f"Stage 1 checkpoint found at {stage1_ckpt} — skipping pre-training.\n")
    else:
        print("=" * 60)
        print("Stage 1: Pre-training on Massachusetts Roads Dataset")
        print("=" * 60)
        train_stage1(STAGE1_CFG)

    # ── Stage 2 + 3 (iterative) ─────────────────────────────────────────────
    cfg = dict(STAGE2_CFG)          # work on a copy so we can mutate lr / resume

    for round_num in range(1, MAX_ROUNDS + 1):
        print("\n" + "=" * 60)
        print(f"Stage 2+3 Round {round_num} / {MAX_ROUNDS}  (lr={cfg['lr']:.2e})")
        print("=" * 60)

        best_auc = train_stage2(cfg)

        print(f"\nRound {round_num} best val AUC-ROC: {best_auc:.4f} (target: {TARGET_AUC})")

        if best_auc >= TARGET_AUC:
            print(f"\nTarget AUC-ROC of {TARGET_AUC} reached after {round_num} round(s).")
            break

        if round_num < MAX_ROUNDS:
            # Continue fine-tuning: resume from best Stage 2 checkpoint, halve LR
            cfg["resume"] = True
            cfg["lr"]     = cfg["lr"] * 0.5
            print(f"Target not reached — continuing fine-tune with lr={cfg['lr']:.2e}")
    else:
        # Load overall best AUC from history to report
        history_path = Path(cfg["history_path"])
        if history_path.exists():
            history = json.loads(history_path.read_text())
            overall_best = max(h.get("val_auc_roc", 0) for h in history)
            print(f"\nMax rounds reached. Overall best val AUC-ROC: {overall_best:.4f}")
            print("Consider adding more labeled images to result.json to improve further.")

    print(f"\nFinal model saved at: {cfg['stage2_ckpt']}")
    print("Run `python quote.py --help` to generate quotes.")


if __name__ == "__main__":
    main()
