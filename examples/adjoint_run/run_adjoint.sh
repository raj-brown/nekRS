#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User configuration
###############################################################################

# Number of samples
N=4

# Epochs / optimizer iterations
NEPOCHS=20

# Interval length
TAU=0.50

# Training start times t_n
TRAIN_TIMES=(0.0 0.5 1.0 1.5)

# MPI launcher
MPI_LAUNCH="mpirun"
NP_FORWARD=4
NP_ADJOINT=4

# NekRS executables
FORWARD_NEKRS="/path/to/nekrs_forward/bin/nekrs"
ADJOINT_NEKRS="/path/to/nekrs_adjoint/bin/nekrs"

# Base case directories
FORWARD_CASE_TEMPLATE="/path/to/cases/forward_case"
ADJOINT_CASE_TEMPLATE="/path/to/cases/adjoint_case"

# Case names used by NekRS
FORWARD_CASE_NAME="channel"
ADJOINT_CASE_NAME="channel_adjoint"

# Training workspace
WORKROOT="$(pwd)/training_workspace"
mkdir -p "$WORKROOT"

# Parameter storage
THETA_FILE="$(pwd)/params/theta.dat"
RMSPROP_STATE="$(pwd)/params/rmsprop_state.dat"

# DNS / filtered DNS data
DNS_DATA_ROOT="/path/to/filtered_dns_data"

# Helper scripts
PREPARE_IC_SCRIPT="$(pwd)/scripts/prepare_initial_condition.sh"
PREPARE_TARGET_SCRIPT="$(pwd)/scripts/prepare_target_state.sh"
COMPUTE_OBJECTIVE_SCRIPT="$(pwd)/scripts/compute_objective.sh"
EXTRACT_GRAD_SCRIPT="$(pwd)/scripts/extract_gradient.sh"
SUM_GRAD_SCRIPT="$(pwd)/scripts/sum_gradients.sh"
RMSPROP_SCRIPT="$(pwd)/scripts/rmsprop_update.sh"

# Optimizer settings
LR="1.0e-3"
BETA="0.9"
EPS="1.0e-8"

###############################################################################
# Sanity checks
###############################################################################

if [[ "${#TRAIN_TIMES[@]}" -ne "$N" ]]; then
  echo "ERROR: TRAIN_TIMES must have N entries"
  exit 1
fi

for f in \
  "$FORWARD_NEKRS" \
  "$ADJOINT_NEKRS" \
  "$PREPARE_IC_SCRIPT" \
  "$PREPARE_TARGET_SCRIPT" \
  "$COMPUTE_OBJECTIVE_SCRIPT" \
  "$EXTRACT_GRAD_SCRIPT" \
  "$SUM_GRAD_SCRIPT" \
  "$RMSPROP_SCRIPT"
do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: required file not found: $f"
    exit 1
  fi
done

###############################################################################
# Functions
###############################################################################

sample_dir() {
  local epoch="$1"
  local n="$2"
  echo "${WORKROOT}/epoch_${epoch}/sample_${n}"
}

prepare_sample() {
  local epoch="$1"
  local n="$2"
  local tn="$3"

  local sdir
  sdir="$(sample_dir "$epoch" "$n")"

  rm -rf "$sdir"
  mkdir -p "$sdir"

  cp -r "$FORWARD_CASE_TEMPLATE" "${sdir}/forward"
  cp -r "$ADJOINT_CASE_TEMPLATE" "${sdir}/adjoint"

  local tend
  tend=$(awk "BEGIN {printf \"%.12g\", $tn + $TAU}")

  cat > "${sdir}/sample.meta" <<EOF
epoch=${epoch}
sample=${n}
t_start=${tn}
tau=${TAU}
t_end=${tend}
EOF
}

write_session_name() {
  local case_dir="$1"
  local case_name="$2"

  cat > "${case_dir}/SESSION.NAME" <<EOF
1
${case_name}
./
EOF
}

configure_forward_par() {
  local case_dir="$1"
  local tn="$2"
  local tend="$3"

  cat > "${case_dir}/${FORWARD_CASE_NAME}.par" <<EOF
[GENERAL]
stopAt = endTime
endTime = ${tend}
dt = 1.0e-3
writeInterval = 0.05
variableDT = no

[RESTART]
restartFromFile = 1
restartFileName = initial_condition.fld

[PRESSURE]
solver = pcg

[VELOCITY]
solver = pcg
EOF
}

configure_adjoint_par() {
  local case_dir="$1"
  local tn="$2"
  local tend="$3"

  cat > "${case_dir}/${ADJOINT_CASE_NAME}.par" <<EOF
[GENERAL]
stopAt = endTime
endTime = ${tn}
dt = 1.0e-3
writeInterval = 0.05
variableDT = no

[RESTART]
restartFromFile = 1
restartFileName = terminal_condition.fld

[PRESSURE]
solver = pcg

[VELOCITY]
solver = pcg
EOF
}

run_forward_sample() {
  local epoch="$1"
  local n="$2"
  local tn="$3"

  local sdir fwd tend
  sdir="$(sample_dir "$epoch" "$n")"
  fwd="${sdir}/forward"
  tend=$(awk "BEGIN {printf \"%.12g\", $tn + $TAU}")

  echo "[epoch ${epoch}][sample ${n}] prepare forward IC from DNS at t=${tn}"
  "$PREPARE_IC_SCRIPT" \
    --dns-root "$DNS_DATA_ROOT" \
    --time "$tn" \
    --theta "$THETA_FILE" \
    --output "${fwd}/initial_condition.fld"

  echo "[epoch ${epoch}][sample ${n}] prepare forward target at t=${tend}"
  "$PREPARE_TARGET_SCRIPT" \
    --dns-root "$DNS_DATA_ROOT" \
    --time "$tend" \
    --output "${fwd}/target_state.fld"

  cp "$THETA_FILE" "${fwd}/theta.dat"

  write_session_name "$fwd" "$FORWARD_CASE_NAME"
  configure_forward_par "$fwd" "$tn" "$tend"

  echo "[epoch ${epoch}][sample ${n}] run forward NekRS"
  (
    cd "$fwd"
    $MPI_LAUNCH -np "$NP_FORWARD" "$FORWARD_NEKRS" > forward.log 2>&1
  )

  echo "[epoch ${epoch}][sample ${n}] compute objective contribution"
  "$COMPUTE_OBJECTIVE_SCRIPT" \
    --pred "${fwd}/final_state.fld" \
    --target "${fwd}/target_state.fld" \
    --output "${sdir}/J_n.dat"

  echo "[epoch ${epoch}][sample ${n}] archive forward data for adjoint"
  mkdir -p "${sdir}/shared"
  cp "${fwd}/final_state.fld" "${sdir}/shared/"
  cp "${fwd}/target_state.fld" "${sdir}/shared/"
  cp "${fwd}/theta.dat" "${sdir}/shared/"

  # If your adjoint needs time history / checkpoints:
  if compgen -G "${fwd}/checkpoints/*" > /dev/null; then
    mkdir -p "${sdir}/shared/checkpoints"
    cp -r "${fwd}/checkpoints/." "${sdir}/shared/checkpoints/"
  fi
}

run_adjoint_sample() {
  local epoch="$1"
  local n="$2"
  local tn="$3"

  local sdir adj tend
  sdir="$(sample_dir "$epoch" "$n")"
  adj="${sdir}/adjoint"
  tend=$(awk "BEGIN {printf \"%.12g\", $tn + $TAU}")

  echo "[epoch ${epoch}][sample ${n}] prepare terminal adjoint condition"
  "$PREPARE_TARGET_SCRIPT" \
    --mode adjoint_terminal \
    --forward-final "${sdir}/shared/final_state.fld" \
    --target "${sdir}/shared/target_state.fld" \
    --output "${adj}/terminal_condition.fld"

  cp "${sdir}/shared/theta.dat" "${adj}/theta.dat"

  if [[ -d "${sdir}/shared/checkpoints" ]]; then
    mkdir -p "${adj}/checkpoints"
    cp -r "${sdir}/shared/checkpoints/." "${adj}/checkpoints/"
  fi

  write_session_name "$adj" "$ADJOINT_CASE_NAME"
  configure_adjoint_par "$adj" "$tn" "$tend"

  echo "[epoch ${epoch}][sample ${n}] run adjoint NekRS"
  (
    cd "$adj"
    $MPI_LAUNCH -np "$NP_ADJOINT" "$ADJOINT_NEKRS" > adjoint.log 2>&1
  )

  echo "[epoch ${epoch}][sample ${n}] extract gradient"
  "$EXTRACT_GRAD_SCRIPT" \
    --adjoint-dir "$adj" \
    --forward-dir "${sdir}/forward" \
    --theta "${adj}/theta.dat" \
    --output "${sdir}/grad_n.dat"
}

sum_objective() {
  local epoch="$1"
  local epoch_dir="${WORKROOT}/epoch_${epoch}"
  : > "${epoch_dir}/objective_values.list"

  local i n sdir
  for ((i=0; i<N; ++i)); do
    n=$((i+1))
    sdir="$(sample_dir "$epoch" "$n")"
    cat "${sdir}/J_n.dat" >> "${epoch_dir}/objective_values.list"
  done

  awk '{s+=$1} END{print s+0.0}' "${epoch_dir}/objective_values.list" \
    > "${epoch_dir}/J_total.dat"

  cat "${epoch_dir}/J_total.dat"
}

sum_gradients() {
  local epoch="$1"
  local epoch_dir="${WORKROOT}/epoch_${epoch}"

  local grad_files=()
  local i n sdir
  for ((i=0; i<N; ++i)); do
    n=$((i+1))
    sdir="$(sample_dir "$epoch" "$n")"
    grad_files+=("${sdir}/grad_n.dat")
  done

  "$SUM_GRAD_SCRIPT" "${grad_files[@]}" > "${epoch_dir}/grad_total.dat"
}

###############################################################################
# Main training loop
###############################################################################

for epoch in $(seq 1 "$NEPOCHS"); do
  echo "============================================================"
  echo "Starting epoch ${epoch}"
  echo "============================================================"

  mkdir -p "${WORKROOT}/epoch_${epoch}"

  # Prepare all sample directories
  for ((i=0; i<N; ++i)); do
    n=$((i+1))
    tn="${TRAIN_TIMES[$i]}"
    prepare_sample "$epoch" "$n" "$tn"
  done

  ###########################################################################
  # Forward solves in parallel
  ###########################################################################
  echo "Launching forward solves..."
  forward_pids=()

  for ((i=0; i<N; ++i)); do
    n=$((i+1))
    tn="${TRAIN_TIMES[$i]}"

    (
      run_forward_sample "$epoch" "$n" "$tn"
    ) &
    forward_pids+=($!)
  done

  for pid in "${forward_pids[@]}"; do
    wait "$pid"
  done

  J_TOTAL="$(sum_objective "$epoch")"
  echo "[epoch ${epoch}] J(theta) = ${J_TOTAL}"

  ###########################################################################
  # Adjoint solves in parallel
  ###########################################################################
  echo "Launching adjoint solves..."
  adjoint_pids=()

  for ((i=0; i<N; ++i)); do
    n=$((i+1))
    tn="${TRAIN_TIMES[$i]}"

    (
      run_adjoint_sample "$epoch" "$n" "$tn"
    ) &
    adjoint_pids+=($!)
  done

  for pid in "${adjoint_pids[@]}"; do
    wait "$pid"
  done

  sum_gradients "$epoch"

  ###########################################################################
  # RMSprop update
  ###########################################################################
  echo "[epoch ${epoch}] update theta with RMSprop"
  "$RMSPROP_SCRIPT" \
    --theta "$THETA_FILE" \
    --grad "${WORKROOT}/epoch_${epoch}/grad_total.dat" \
    --state "$RMSPROP_STATE" \
    --lr "$LR" \
    --beta "$BETA" \
    --eps "$EPS"

  echo "[epoch ${epoch}] completed"
done

echo "Training finished."
