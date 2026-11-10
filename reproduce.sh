#!/usr/bin/env bash
# The exact runs behind every main-paper number, grouped by figure / table. Each line is one cell of 50 paired test
# episodes (policy seed 7); the paper's count is given in the comment. Usage: bash reproduce.sh GROUP [GROUP ...]
#   groups: fig2 fig4 fig6 tab2 fig8a fig8b heldout   (or "list" to print the commands without running them)
# Each run needs one GPU (policy server) and a few CPU cores (simulator). Runs are independent: to use several GPUs,
# run groups in parallel with different CUDA_VISIBLE_DEVICES, or copy single lines. Extra run_eval.sh options (e.g.
# --render lavapipe, the renderer used for the paper's numbers) can be passed in RUN_EVAL_ARGS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DRY=0
run() { if [ $DRY == 1 ]; then echo "bash run_eval.sh $* ${RUN_EVAL_ARGS:-}"; else bash run_eval.sh "$@" ${RUN_EVAL_ARGS:-}; fi; }

fig2() {  # Fig. 2 / Tab. 1 -- null events break the released FrameSamp-Modul policy, only when they reach memory
  for task in PickXtimes SwingXtimes; do
    run --task $task --event none                                      # Pick 47 | Swing 44
    run --task $task --event pause --len 100 --memory visible          # Pick 16 | Swing 10
    run --task $task --event pause --len 100 --memory hidden           # Pick 45 | Swing 47
    run --task $task --event fidget --len 100 --memory visible         # Pick 16 | Swing 10
    run --task $task --event fidget --len 100 --memory hidden          # Pick 46 | Swing 45
  done
  run --task SwingXtimes --event airgrasp-return --memory visible      # 23
  run --task SwingXtimes --event airgrasp-return --memory hidden       # 47
  # Tab. 1 (successes by K) is the same runs split by the number of repetitions in each task goal.
}

fig4() {  # Fig. 4 -- rescaling the time stamps (no event) shifts the failure: s < 1 extra repetitions, s > 1 early presses
  for s in 0.5 0.6 0.7 0.75 1.1 1.25 1.5; do
    run --task PickXtimes --clock-scale $s     # 0.5: 24 | 0.6: 33 | 0.7: 40 | 0.75: 45 | (1: 47) | 1.1: 38 | 1.25: 30 | 1.5: 18
    run --task SwingXtimes --clock-scale $s    # 0.5: 20 | 0.6: 32 | 0.7: 40 | 0.75: 48 | (1: 44) | 1.1: 48 | 1.25: 43 | 1.5: 37
  done
}

fig6() {  # Fig. 6 -- Clock vs NoClock (trained from pi0.5 base); controls and null-slot dose on NoClock
  for m in clock noclock; do
    for task in PickXtimes SwingXtimes; do
      run --model $m --task $task --event none                         # Clock 41 / 48 | NoClock 42 / 47 (Pick / Swing)
      run --model $m --task $task --event pause --len 100              # Clock 17 / 10 | NoClock 21 / 14
      run --model $m --task $task --event pause --len 100 --memory hidden   # Clock 38 / 38 | NoClock 44 / 46
    done
    run --model $m --task PickXtimes --clock-scale 1.5                 # Clock 19 | NoClock 44
    run --model $m --task PickXtimes --event fidget --len 100          # Clock 16 | NoClock 23
    run --model $m --task PickXtimes --control dupfill --sim-pause 100 # Clock 39 | NoClock 38
  done
  run --model noclock --task SwingXtimes --control dupfill --sim-pause 100          # 48
  for task in PickXtimes SwingXtimes; do
    run --model noclock --task $task --event pause --len 100 --control nullswap     # Pick 17 | Swing 13
    for m in 1 3 6; do
      run --model noclock --task $task --control nullblock --nb-m $m   # Pick 45 / 37 / 24 | Swing 47 / 43 / 23 (m = 1 / 3 / 6)
    done
  done
}

tab2() {  # Tab. 2 -- other released RoboMME designs; pauses of 300 steps, fidgets of 100
  for m in perceptual-framesamp-modul perceptual-framesamp-expert perceptual-framesamp-context perceptual-tokendrop-modul \
           memer text-subgoal recurrent-ttt-context pi05_baseline; do
    run --model $m --task PickXtimes --event none
    run --model $m --task PickXtimes --event pause --len 300
    run --model $m --task PickXtimes --event fidget --len 100
    run --model $m --task SwingXtimes --event none
    run --model $m --task SwingXtimes --event pause --len 300
  done
  # Pick none -> pause / fidget, Swing none -> pause:
  #   FrameSamp-Modul 47 -> 16 / 16, 44 -> 12      FrameSamp-Expert 44 -> 14 / 17, 49 -> 11
  #   FrameSamp-Context 33 -> 11 / 20, 43 -> 10    TokenDrop-Modul 39 -> 30 / 21, 45 -> 27
  #   MemER-style 38 -> 39 / 38, 24 -> 23          text sub-goals 46 -> 48 / 47, 1 -> 2
  #   TTT 23 -> 22 / 25, 17 -> 16                  pi0.5 18 -> 19 / 20, 15 -> 13
}

fig8a() {  # Fig. 8a -- pause augmentation vs a retrain of the Clock recipe on the same machine
  for m in clock-retrain pauseaug pauseaug-weak; do
    for task in PickXtimes SwingXtimes; do
      run --model $m --task $task --event none                         # retrain 32 / 46 | aug 24 / 19 | weak 39 / 48
      run --model $m --task $task --event pause --len 100              # retrain 16 / 11 | aug 25 / 10 | weak 19 / 12
    done
    run --model $m --task PickXtimes --event fidget --len 100          # aug 21 | weak 18
    run --model $m --task SwingXtimes --event airgrasp-return          # aug 17 | weak 24
  done
}

fig8b() {  # Fig. 8b -- the event index at test time on the Clock checkpoint (compare with fig6's Clock runs)
  for task in PickXtimes SwingXtimes; do
    run --model clock --task $task --event none --index event          # Pick 36 | Swing 48
    run --model clock --task $task --event pause --len 100 --index event   # Pick 34 | Swing 30
  done
  run --model clock --task PickXtimes --event fidget --len 100 --index event   # 33 (without the index: 16)
  run --model clock --task SwingXtimes --event airgrasp-return --index event   # 43 (without the index: 13)
  run --model clock --task SwingXtimes --event airgrasp-return                 # 13
}

heldout() {  # held-out replication of Fig. 8b on the val split (episodes never used for any design decision)
  for task in PickXtimes SwingXtimes; do
    run --model clock --task $task --split val --event none                             # Pick 38 | Swing 44
    run --model clock --task $task --split val --event pause --len 100                  # Pick 10 | Swing 14
    run --model clock --task $task --split val --event pause --len 100 --index event    # Pick 36 | Swing 32
  done
  run --model clock --task PickXtimes --split val --event none --index event            # 40
}

[ $# -gt 0 ] || { sed -n '2,8p' "$0"; exit 1; }
for g in "$@"; do
  case $g in
    list) DRY=1; for f in fig2 fig4 fig6 tab2 fig8a fig8b heldout; do echo "# $f"; $f; done;;
    fig2|fig4|fig6|tab2|fig8a|fig8b|heldout) $g;;
    *) echo "unknown group $g"; exit 1;;
  esac
done
