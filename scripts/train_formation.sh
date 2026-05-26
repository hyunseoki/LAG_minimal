#!/bin/bash
# Train wedge formation flight with MAPPO algorithm
# Leader flies ㄱ-shaped path, followers maintain wedge formation

set -e

# Configuration
ENV_NAME="Formation"
SCENARIO_NAME="formation/wedge"
ALGORITHM_NAME="mappo"
EXPERIMENT_NAME="v1"
SEED=5

# Training parameters
NUM_TRAINING_THREADS=1
NUM_ROLLOUT_THREADS=4
NUM_MINI_BATCH=5
BUFFER_SIZE=3000
NUM_ENV_STEPS=1e8

# Optional: use GPU
GPU=0

# Optional: enable WandB logging
USE_WANDB=false
WANDB_NAME="formation_wedge_${SEED}"
USER_NAME="your_username"

# Run training
CUDA_VISIBLE_DEVICES=$GPU python train_jsbsim.py \
    --env-name $ENV_NAME \
    --algorithm-name $ALGORITHM_NAME \
    --scenario-name $SCENARIO_NAME \
    --experiment-name $EXPERIMENT_NAME \
    --seed $SEED \
    --n-training-threads $NUM_TRAINING_THREADS \
    --n-rollout-threads $NUM_ROLLOUT_THREADS \
    --use-wandb \
    --log-interval 1 \
    --save-interval 1 \
    --num-mini-batch $NUM_MINI_BATCH \
    --buffer-size $BUFFER_SIZE \
    --num-env-steps $NUM_ENV_STEPS
    # --wandb-name $WANDB_NAME \
    # --user-name $USER_NAME

echo "Training completed!"
