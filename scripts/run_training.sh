#!/bin/bash
#
# Run Training Script - Parallel execution of master_pipeline.py on multiple objects
#
# The master_pipeline.py trains ONE object at a time across ALL selected pipelines.
# This maximizes YOLO cache efficiency (same object's bboxes reused across pipelines).
#
# This script launches multiple parallel instances, each training a different object.
#
# Usage:
#   ./scripts/run_training.sh                    # Train all objects sequentially
#   ./scripts/run_training.sh -j 2               # Train 2 objects in parallel
#   ./scripts/run_training.sh -j 4 -p rgb        # Train RGB pipeline, 4 parallel
#   ./scripts/run_training.sh -o "ape cat duck"  # Train specific objects
#
# Arguments:
#   -j, --jobs      Number of parallel jobs (default: 1)
#   -o, --objects   Space-separated list of objects (default: all)
#   -p, --pipelines Space-separated list of pipelines (default: all)
#   -e, --epochs    Number of epochs for both RGB and RGB-D (overrides defaults)
#   -h, --help      Show this help message
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# All available objects
ALL_OBJECTS="ape benchvise camera can cat driller duck eggbox glue holepuncher iron lamp phone"

# All available pipelines
ALL_PIPELINES="rgb residual_learning densefusion_iterative pvn3d ffb6d"

# Default values
PARALLEL_JOBS=1
OBJECTS="$ALL_OBJECTS"
PIPELINES="$ALL_PIPELINES"
EPOCHS=""

# =============================================================================
# Parse Arguments
# =============================================================================

show_help() {
    cat << EOF
Run Training Script - Parallel execution of master_pipeline.py

Each job trains ONE object across ALL selected pipelines (YOLO cache efficient).

Usage: $(basename "$0") [OPTIONS]

Options:
    -j, --jobs NUM       Number of parallel training jobs (default: 1)
                         Each job trains one object with all pipelines
    -o, --objects LIST   Space-separated list of objects to train (default: all)
                         Available: $ALL_OBJECTS
    -p, --pipelines LIST Space-separated list of pipelines to train (default: all)
                         Available: $ALL_PIPELINES
    -e, --epochs NUM     Number of epochs for both RGB and RGB-D
    -h, --help           Show this help message

Examples:
    $(basename "$0")                           # Train all objects/pipelines sequentially
    $(basename "$0") -j 2                      # Train 2 objects in parallel
    $(basename "$0") -j 4 -p "rgb"             # RGB pipeline only, 4 parallel jobs
    $(basename "$0") -o "ape cat duck" -j 3    # Train 3 specific objects in parallel
    $(basename "$0") -p "residual_learning" -e 50  # Train residual_learning for 50 epochs

Output:
    Logs are saved to: ${LOG_DIR}/
    Each object creates: <object>_<timestamp>.log
    Summary log: training_summary_<timestamp>.log
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -o|--objects)
            OBJECTS="$2"
            shift 2
            ;;
        -p|--pipelines)
            PIPELINES="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

# Create log directory
mkdir -p "$LOG_DIR"

# Summary log file
SUMMARY_LOG="${LOG_DIR}/training_summary_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

# =============================================================================
# Main
# =============================================================================

log_message "=========================================="
log_message "TRAINING SCRIPT STARTED"
log_message "=========================================="
log_message "Project root: $PROJECT_ROOT"
log_message "Log directory: $LOG_DIR"
log_message "Parallel jobs: $PARALLEL_JOBS"
log_message "Objects: $OBJECTS"
log_message "Pipelines: $PIPELINES"
if [[ -n "$EPOCHS" ]]; then
    log_message "Epochs: $EPOCHS"
fi
log_message "=========================================="

# Convert objects and pipelines to arrays
read -ra OBJECT_ARRAY <<< "$OBJECTS"
read -ra PIPELINE_ARRAY <<< "$PIPELINES"

# Total tasks = number of objects (each trains all pipelines)
TOTAL_TASKS=${#OBJECT_ARRAY[@]}
log_message "Total objects to train: $TOTAL_TASKS"
log_message "Each object trains ${#PIPELINE_ARRAY[@]} pipeline(s)"

# Build extra args for pipelines and epochs
PIPELINE_ARGS="--pipelines ${PIPELINES}"
EXTRA_ARGS=""
if [[ -n "$EPOCHS" ]]; then
    EXTRA_ARGS="--epochs_rgb $EPOCHS --epochs_rgbd $EPOCHS"
fi

# Function to run training for a single object (all pipelines)
run_training_job() {
    local object="$1"
    local job_log="${LOG_DIR}/${object}_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $object (log: $job_log)"
    
    # Run the training for this object with all selected pipelines
    cd "$PROJECT_ROOT"
    python scripts/master_pipeline.py \
        --objects "$object" \
        $PIPELINE_ARGS \
        $EXTRA_ARGS \
        > "$job_log" 2>&1
    
    local status=$?
    
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $object (SUCCESS)"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $object (FAILED - see $job_log)"
        return 1
    fi
}

export -f run_training_job
export LOG_DIR TIMESTAMP PROJECT_ROOT PIPELINE_ARGS EXTRA_ARGS

# Track start time
START_TIME=$(date +%s)

# Build object list
OBJECT_LIST=$(printf '%s\n' "${OBJECT_ARRAY[@]}")

# Run tasks with parallel or xargs
log_message "Starting training with $PARALLEL_JOBS parallel job(s)..."

# Check if GNU parallel is available
if command -v parallel &> /dev/null && [[ $PARALLEL_JOBS -gt 1 ]]; then
    # Use GNU parallel
    log_message "Using GNU parallel for job management"
    
    echo "$OBJECT_LIST" | parallel -j "$PARALLEL_JOBS" \
        "run_training_job {}" \
        2>&1 | tee -a "$SUMMARY_LOG"
else
    # Fallback to background jobs with manual control
    log_message "Using bash background jobs for parallel execution"
    
    RUNNING_JOBS=0
    PIDS=()
    TASKS=()
    
    for object in "${OBJECT_ARRAY[@]}"; do
        # Wait if we've reached the max parallel jobs
        while [[ $RUNNING_JOBS -ge $PARALLEL_JOBS ]]; do
            # Wait for any job to finish
            for i in "${!PIDS[@]}"; do
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    wait "${PIDS[$i]}" 2>/dev/null || true
                    unset 'PIDS[i]'
                    unset 'TASKS[i]'
                    ((RUNNING_JOBS--)) || true
                fi
            done
            # Re-index arrays
            PIDS=("${PIDS[@]}")
            TASKS=("${TASKS[@]}")
            sleep 1
        done
        
        # Start new job in background
        run_training_job "$object" &
        PIDS+=($!)
        TASKS+=("$object")
        ((RUNNING_JOBS++)) || true
    done
    
    # Wait for remaining jobs
    log_message "Waiting for remaining jobs to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

log_message "=========================================="
log_message "TRAINING COMPLETED"
log_message "=========================================="
log_message "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_message "Summary log: $SUMMARY_LOG"
log_message "Individual logs: ${LOG_DIR}/<object>_${TIMESTAMP}.log"

# Print summary of results
log_message ""
log_message "Results summary:"
SUCCESS_COUNT=0
FAIL_COUNT=0

for object in "${OBJECT_ARRAY[@]}"; do
    job_log="${LOG_DIR}/${object}_${TIMESTAMP}.log"
    if [[ -f "$job_log" ]] && grep -q "TRAINING COMPLETE" "$job_log" 2>/dev/null; then
        log_message "  ✓ ${object}: SUCCESS"
        ((SUCCESS_COUNT++)) || true
    else
        log_message "  ✗ ${object}: FAILED"
        ((FAIL_COUNT++)) || true
    fi
done

log_message ""
log_message "Success: $SUCCESS_COUNT / $TOTAL_TASKS objects"
log_message "Failed:  $FAIL_COUNT / $TOTAL_TASKS objects"
log_message "=========================================="

exit 0
