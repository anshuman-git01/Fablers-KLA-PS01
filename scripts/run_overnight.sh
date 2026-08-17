#!/bin/bash
# Unattended overnight chain.
#
#   1. wait for the in-flight unet_l1_b48 run to finish (does not touch that process)
#   2. train base32 U-Net with the combined L1+SSIM+LPIPS loss, otherwise identical settings
#      to unet_l1_b32 so it stays a single-variable comparison
#   3. generate evaluation reports for both the b48 run and the new loss run
#
# Every step appends to results/runs/overnight.log with timestamps. Steps are independent:
# a failure in one is logged and the chain continues, so a late failure cannot wipe out
# earlier completed work.

cd /Users/anshuman/kla-restoration || exit 1

LOG=results/runs/overnight.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== overnight chain started (pid $$) ==="

# --- 1. wait for the running b48 job ---------------------------------------------------------
if pgrep -f "train.py --name unet_l1_b48" > /dev/null; then
    say "waiting for unet_l1_b48 to finish..."
    while pgrep -f "train.py --name unet_l1_b48" > /dev/null; do sleep 60; done
    say "unet_l1_b48 finished"
else
    say "unet_l1_b48 not running; continuing"
fi
sleep 10

# --- 2. combined-loss experiment -------------------------------------------------------------
say "starting unet_ssimlpips_b32 (L1+SSIM+LPIPS, base32, 60 epochs)"
python3 -u train.py \
    --name unet_ssimlpips_b32 \
    --arch unet --base 32 --blocks-per-level 2 \
    --loss l1_ssim_lpips --w-l1 1.0 --w-ssim 0.15 --w-lpips 0.10 \
    --epochs 60 --batch-size 16 --lr 5e-4 --seed 42 \
    > results/runs/unet_ssimlpips_b32.log 2>&1
say "unet_ssimlpips_b32 exited with code $?"

# --- 3. evaluation reports -------------------------------------------------------------------
for ck in weights/unet_l1_b48_best.pt weights/unet_ssimlpips_b32_best.pt; do
    if [ -f "$ck" ]; then
        say "eval_report on $ck"
        python3 scripts/eval_report.py --checkpoint "$ck" >> "$LOG" 2>&1
        say "  -> exit $?"
    else
        say "SKIP eval_report: $ck not found"
    fi
done

say "=== overnight chain complete ==="
