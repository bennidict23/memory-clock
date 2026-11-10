#!/usr/bin/env bash
# Evaluate one condition: start a policy server on a free port, wait for it, run the RoboMME client against it, stop
# the server, print the number of successes. One server serves one run (the memory lives in the server).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOMME_DIR=${ROBOMME_DIR:-$REPO/third_party/robomme_policy_learning}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/checkpoints/openpi}

usage() {
  cat <<'EOF'
Usage: bash run_eval.sh [options]
  --model NAME        released RoboMME model (default perceptual-framesamp-modul; also perceptual-framesamp-expert,
                      perceptual-framesamp-context, perceptual-tokendrop-modul, recurrent-ttt-context, pi05_baseline),
                      the paper's trained models clock | noclock | clock-retrain | pauseaug | pauseaug-weak
                      (checkpoints/<name>/29999),
                      memer (MemER-style keyframes) or text-subgoal (text sub-goals); with --ckpt, any label
  --ckpt DIR          custom checkpoint directory (with params/ and assets/; its parent holds history_config.txt)
  --task NAME         RoboMME task (default PickXtimes)
  --event E           none | pause | fidget | airgrasp | airgrasp-return (default none)
  --len N             pause / fidget length in env steps (default 100)
  --at LIST           policy steps after which the events are injected (default 48,160)
  --memory M          visible | hidden: whether event frames enter memory (default visible)
  --index I           time | event: time-indexed memory (stock) or the test-time event index (default time)
  --clock-scale S     multiply the time stamps of all memory frames by S (no event needed)
  --control C         nullswap (needs a visible event) | dupfill | nullblock (both without event)
  --sim-pause P       dupfill: length of the virtual pauses (default 100)
  --sim-at LIST       dupfill / nullblock: virtual event times (default: the value of --at)
  --nb-m M            nullblock: identical slots per virtual event (default 6)
  --split S           test | val (default test)
  --episodes N        episodes per task (default 50)
  --seed N            policy noise seed (default 7)
  --render R          default | lavapipe (CPU Vulkan rendering, see memory_clock/render_lavapipe.py)
  --out DIR           results root (default ./results); results go to <out>/<tag>/
  --tag NAME          result tag (default: built from the options)
  --port N            server port (default: a free one)
  --overwrite         discard earlier results of the same tag (default: resume from progress.json)
Environment: ROBOMME_DIR (robomme_policy_learning checkout, default third_party/robomme_policy_learning),
  OPENPI_DATA_HOME, CUDA_VISIBLE_DEVICES, XLA_PYTHON_CLIENT_MEM_FRACTION (default 0.85; 0.45 with a VLM sub-goal
  predictor, which runs in the client on the same GPU), LP_NUM_THREADS (lavapipe render threads).
EOF
}

MODEL=perceptual-framesamp-modul; CKPT=""; TASK=PickXtimes; EVENT=none; LEN=100; AT=48,160; MEMORY=visible
INDEX=time; CLOCK_SCALE=""; CONTROL=""; SIM_PAUSE=100; SIM_AT=""; NB_M=6; SPLIT=test; EPISODES=50; SEED=7
RENDER=default; OUT=$REPO/results; TAG=""; PORT=""; OVERWRITE=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2;; --ckpt) CKPT=$2;; --task) TASK=$2;; --event) EVENT=$2;; --len) LEN=$2;; --at) AT=$2;;
    --memory) MEMORY=$2;; --index) INDEX=$2;; --clock-scale) CLOCK_SCALE=$2;; --control) CONTROL=$2;;
    --sim-pause) SIM_PAUSE=$2;; --sim-at) SIM_AT=$2;; --nb-m) NB_M=$2;; --split) SPLIT=$2;; --episodes) EPISODES=$2;;
    --seed) SEED=$2;; --render) RENDER=$2;; --out) OUT=$2;; --tag) TAG=$2;; --port) PORT=$2;;
    --overwrite) OVERWRITE=1; shift; continue;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1"; usage; exit 1;;
  esac
  [ $# -ge 2 ] || { echo "option $1 needs a value"; exit 1; }
  shift 2
done
SIM_AT=${SIM_AT:-$AT}
die() { echo "run_eval.sh: $*" >&2; exit 1; }
[ -d "$ROBOMME_DIR/.venv" ] || die "no robomme_policy_learning checkout with .venv at $ROBOMME_DIR (run install.sh)"
ROBOMME_DIR=$(cd "$ROBOMME_DIR" && pwd)
case $EVENT in none|pause|fidget|airgrasp|airgrasp-return) ;; *) die "--event $EVENT";; esac
case $MEMORY in visible|hidden) ;; *) die "--memory $MEMORY";; esac
case $INDEX in time|event) ;; *) die "--index $INDEX";; esac
case $CONTROL in ""|nullswap|dupfill|nullblock) ;; *) die "--control $CONTROL";; esac
case $SPLIT in test|val) ;; *) die "--split $SPLIT";; esac
case $RENDER in default|lavapipe) ;; *) die "--render $RENDER";; esac
N_KNOBS=0
[ -n "$CLOCK_SCALE" ] && N_KNOBS=$((N_KNOBS + 1))
[ -n "$CONTROL" ] && N_KNOBS=$((N_KNOBS + 1))
[ $INDEX == event ] && N_KNOBS=$((N_KNOBS + 1))
[ $N_KNOBS -le 1 ] || die "use at most one of --clock-scale, --control, --index event"
[ "$CONTROL" == nullswap ] && { [ $EVENT == none ] || [ $MEMORY != visible ]; } && die "nullswap needs a visible event"
case $CONTROL in dupfill|nullblock) [ $EVENT == none ] || die "--control $CONTROL is run without events";; esac

# memory manipulations on the server
MEM_MODE=stock
[ -n "$CLOCK_SCALE" ] && MEM_MODE=clock_scale
[ -n "$CONTROL" ] && MEM_MODE=$CONTROL
[ $INDEX == event ] && MEM_MODE=event_index

# model -> checkpoint, train config, client options
MODEL_TYPE=$MODEL; CONFIG=mme_vla_suite; EXTRA=(); SUBGOAL=""
case $MODEL in
  memer) MODEL_TYPE=symbolic-grounded-subgoal; SUBGOAL=memer; EXTRA=(--args.subgoal-type=grounded_subgoal --args.use-memer);;
  text-subgoal) MODEL_TYPE=symbolic-simple-subgoal; SUBGOAL=qwenvl_simple; EXTRA=(--args.subgoal-type=simple_subgoal --args.use-qwenvl);;
  pi05_baseline) CONFIG=pi05_baseline; EXTRA=(--args.no-use-history);;
esac
if [ -n "$CKPT" ]; then CKPT_DIR=$(cd "$CKPT" && pwd)
else case $MODEL in
  clock|noclock|clock-retrain|pauseaug|pauseaug-weak) CKPT_DIR=$REPO/checkpoints/$MODEL/29999;;
  pi05_baseline) CKPT_DIR=$ROBOMME_DIR/runs/ckpts/pi05_baseline/pi05_baseline/79999;;
  *) CKPT_DIR=$ROBOMME_DIR/runs/ckpts/mme_vla_suite/$MODEL_TYPE/79999;;
esac; fi
[ -d "$CKPT_DIR/params" ] || die "no checkpoint at $CKPT_DIR (see download.sh)"
# Which knobs are meaningful: pi0.5 has no memory, symbolic models store text, TokenDrop / recurrent memories do not
# sample 32 frames (only the time stamps can be rescaled there).
case $MODEL_TYPE in
  pi05_baseline|symbolic-*) [ $MEM_MODE == stock ] || die "$MODEL has no frame memory to manipulate";;
  *tokendrop*|*ttt*|*recurrent*) case $MEM_MODE in stock|clock_scale) ;; *) die "$MEM_MODE is not defined for $MODEL";; esac;;
esac
VLM_DIR=""
case $SUBGOAL in qwenvl_*|memer) VLM_DIR=$ROBOMME_DIR/runs/ckpts/Qwen3-VL-4B-Instruct; DEF_FRAC=0.45;; *) DEF_FRAC=0.85;; esac

if [ -z "$TAG" ]; then
  case $EVENT in none) EV=noevent;; pause|fidget) EV=$EVENT$LEN-$MEMORY;; *) EV=$EVENT-$MEMORY;; esac
  [ "$EVENT" != none ] && [ "$AT" != 48,160 ] && EV=$EV-at${AT//,/_}
  TAG=${MODEL}__${TASK}__$EV
  [ $INDEX == event ] && TAG=$TAG-eventindex
  [ -n "$CLOCK_SCALE" ] && TAG=$TAG-clock$CLOCK_SCALE
  case $CONTROL in
    nullswap) TAG=$TAG-nullswap;;
    dupfill) TAG=$TAG-dupfill$SIM_PAUSE;;
    nullblock) TAG=$TAG-nullblock$NB_M;;
  esac
  case $CONTROL in dupfill|nullblock) [ "$SIM_AT" != 48,160 ] && TAG=$TAG-at${SIM_AT//,/_};; esac
  [ $SPLIT == val ] && TAG=$TAG-val
  [ "$SEED" != 7 ] && TAG=$TAG-seed$SEED
  [ $RENDER == lavapipe ] && TAG=$TAG-lvp
fi
mkdir -p "$OUT/$TAG"; OUT=$(cd "$OUT" && pwd); LOG=$OUT/$TAG
PORT=${PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
port_open() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1', $PORT)))"; }
# a busy port would connect the client to another run's server, whose memory would then mix two episodes
port_open && die "port $PORT is already in use"
echo "[run_eval] $TAG: $MODEL on $TASK ($SPLIT, $EPISODES episodes, seed $SEED), event $EVENT, memory $MEMORY, server $MEM_MODE, port $PORT"

cd "$ROBOMME_DIR"
MEM_MODE=$MEM_MODE CLOCK_SCALE=${CLOCK_SCALE:-1.0} SIM_PAUSE=$SIM_PAUSE SIM_AT=$SIM_AT NB_M=$NB_M ROBOMME_DIR=$ROBOMME_DIR \
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$DEF_FRAC} \
  .venv/bin/python "$REPO/memory_clock/policy_server.py" --seed="$SEED" --port="$PORT" \
  policy:checkpoint --policy.dir="$CKPT_DIR" --policy.config=$CONFIG > "$LOG/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  port_open && sleep 2 && kill -0 $SERVER 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -20 "$LOG/server.log"; die "policy server died (log: $LOG/server.log)"; }
  [ "$i" == 120 ] && die "policy server did not come up in 10 min (log: $LOG/server.log)"
  sleep 5
done
echo "[run_eval] server up; client log: $LOG/client.log"

RENDER_ENV=()
if [ $RENDER == lavapipe ]; then
  RENDER_ENV=(VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}
              LP_NUM_THREADS=${LP_NUM_THREADS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))})
fi
[ $OVERWRITE == 1 ] && EXTRA+=(--args.overwrite)
set +e
env ${RENDER_ENV[@]+"${RENDER_ENV[@]}"} EVENT=$EVENT EVENT_LEN=$LEN EVENT_AT=$AT EVENT_MEMORY=$MEMORY INDEX=$INDEX SPLIT=$SPLIT \
  MAX_EPISODES=$EPISODES SUBGOAL=$SUBGOAL VLM_DIR=$VLM_DIR RENDER=$RENDER ROBOMME_DIR=$ROBOMME_DIR \
  .venv-robomme/bin/python -u "$REPO/memory_clock/eval_client.py" --args.model-seed="$SEED" --args.port="$PORT" \
  --args.policy-name="$TAG" --args.model-ckpt-id=79999 --args.only-tasks="$TASK" --args.save-dir="$OUT" ${EXTRA[@]+"${EXTRA[@]}"} \
  > "$LOG/client.log" 2>&1
STATUS=$?
set -e
PROGRESS=$(find "$LOG" -name progress.json | head -1)
[ -n "$PROGRESS" ] || { tail -20 "$LOG/client.log"; die "client exited ($STATUS) without results"; }
python3 - "$PROGRESS" "$TASK" "$TAG" <<'EOF'
import json, sys
eps = json.load(open(sys.argv[1])).get(sys.argv[2], {})
ok = sum(v is True for v in eps.values())
print(f"[run_eval] {sys.argv[3]}: {ok}/{len(eps)} successes ({sys.argv[1].rsplit('/', 1)[0]})")
EOF
exit $STATUS
