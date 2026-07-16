#!/bin/bash

# GSPO Multi-node Training Script for Qwen3-Omni
# Adapted from single-node train_lora_single.sh for distributed training

# Step1: Initialize conda and activate environment
# Please configure your conda environment path
# source /path/to/your/anaconda3/etc/profile.d/conda.sh
# conda activate /path/to/your/env/swift

# Step2: Set proxy (if needed for downloading models or dependencies)
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port

# ------------------------------ Environment Configuration ----------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=0
export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL  # Commented out to reduce log output
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1  # Disabled as it causes deadlock in multi-node environment

# Increase distributed training timeout settings (to resolve TCPStore communication issues)
export TORCH_DISTRIBUTED_INIT_TIMEOUT=7200  # 2 hours timeout for initialization
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1  # Enable blocking wait for better error reporting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Audio sampling rate setting (Qwen3-Omni uses 16000 Hz by default)
export SAMPLING_RATE=16000  # Enable async error handling

# ============== Performance Optimization Settings ==============
# Fix slow training issue with conv3d operator in PyTorch 2.9 (ms-swift>=3.11.2)
# export SWIFT_PATCH_CONV3D=1

# Video processing backend setting - use torchcodec instead of decord to avoid training hangs
# ms-swift uses get_env_args('video_load_backend', str, 'pyav') to read this env variable
# Reference: swift/llm/template/vision_utils.py:197
# export VIDEO_LOAD_BACKEND=torchcodec  # Options: pyav(default), decord, torchcodec
# ==========================================

# Global experiment name variable
EXPERIMENT_NAME="Omnivideo-R1-MA"

# Get number of nodes
nnodes=$HOST_NUM

# Create log directory
LOG_DIR="./logs/gspo"
mkdir -p ${LOG_DIR}

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/gspo_node${INDEX}_${TIMESTAMP}.log"

echo "=========================================="
echo "Node: ${INDEX}/${nnodes}"
echo "Master: ${CHIEF_IP}"
echo "Master Port: 29500"
echo "NPROC_PER_NODE: 8"
echo "Log file: ${LOG_FILE}"
echo "Start time: $(date)"
echo "=========================================="

# Print network interface information
echo "Network interface bond1 status:"
ip addr show bond1 || echo "bond1 interface not found"
echo ""

# If this is a Worker node, wait and test connection to Master node
if [ ${INDEX} -ne 0 ]; then
    echo "🔍 This is Worker node, testing connection to Master node ${CHIEF_IP}:29500..."
    
    RETRY_COUNT=0
    MAX_RETRIES=30
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if timeout 5 bash -c "echo >/dev/tcp/${CHIEF_IP}/29500" 2>/dev/null; then
            echo "✓ Successfully connected to Master node!"
            break
        else
            echo "⏳ Waiting for Master node... Attempt $((RETRY_COUNT+1))/${MAX_RETRIES}"
            sleep 10
            ((RETRY_COUNT++))
        fi
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "❌ ERROR: Cannot connect to Master node after ${MAX_RETRIES} attempts!"
        echo "Please check:"
        echo "  1. Master node is running"
        echo "  2. Port 29500 is open on ${CHIEF_IP}"
        echo "  3. Network connectivity between nodes"
        exit 1
    fi
else
    echo "📡 This is Master node (INDEX=0), will create TCPStore server"
fi

echo ""

# GSPO Training with Multi-node Configuration
# GSPO hyperparameters from paper https://arxiv.org/pdf/2507.18071
# - epsilon = 3e-4 from paper section 5.1
# - epsilon_high = 4e-4 from paper section 5.1
# - steps_per_generation = 4 from paper section 5.1
# - beta = 0: zero kl regularization

# Video sampling parameters to reduce token count
FPS_MAX_FRAMES=64 \
MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NNODES=$nnodes \
NODE_RANK=$INDEX \
MASTER_ADDR=$CHIEF_IP \
MASTER_PORT=29500 \
megatron rlhf \
  --rlhf_type grpo \
  --model /path/to/your/models/Qwen3-Omni-30B-A3B-Instruct \
  --model_type qwen3_omni \
  --save ./output/gspo_output/${EXPERIMENT_NAME} \
  --add_version false \
  --tensorboard_dir ./output/gspo_output/${EXPERIMENT_NAME}/runs \
  --wandb_save_dir ./output/gspo_output/${EXPERIMENT_NAME} \
  --load_safetensors true \
  --save_safetensors true \
  --context_parallel_size 1 \
  --tensor_model_parallel_size 4 \
  --expert_model_parallel_size 4 \
  --pipeline_model_parallel_size 2 \
  --dataset ./data/merged_train_fusion_ma.jsonl \
  --max_epochs 1 \
  --global_batch_size 112 \
  --micro_batch_size 1 \
  --steps_per_generation 1 \
  --num_generations 8 \
  --num_iterations 1 \
  --beta 0.03 \
  --importance_sampling_level sequence \
  --epsilon 3e-4 \
  --epsilon_high 4e-4 \
  --dynamic_sample false \
  --overlong_filter true \
  --loss_type grpo \
  --external_plugins ./grpo/plugin/omnivideo_r1.py \
  --reward_funcs soft_mc_acc tcta_format modality_attention \
  --reward_weights 1 1 1 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_tensor_parallel_size 8 \
  --vllm_max_model_len 32768 \
  --max_length 32768 \
  --max_completion_length 8192 \
  --train_type lora \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_parameters talker code2wav \
  --lr 1e-6 \
  --lr_warmup_fraction 0.05 \
  --min_lr 1e-7 \
  --bf16 true \
  --use_precision_aware_optimizer \
  --moe_permute_fusion true \
  --moe_grouped_gemm true \
  --moe_shared_expert_overlap true \
  --moe_aux_loss_coeff 1e-3 \
  --sleep_level 0 \
  --offload_model true \
  --load_from_cache_file true \
  --recompute_granularity selective \
  --vit_gradient_checkpointing true \
  --padding_free true \
  --sequence_parallel true \
  --save_interval 50 \
  --no_save_optim true \
  --no_save_rng true \
  --log_interval 1 \
  --num_workers 32 \
  --dataset_num_proc 32 \
  --attention_backend flash \
  --temperature 1.0 \
  --torch_dtype bfloat16 \
  --no_gradient_accumulation_fusion true \
  --system './prompt.txt' \
  2>&1 | tee -a ${LOG_FILE}

# Capture the exit status of the training command
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Record training end status
echo "==========================================" | tee -a ${LOG_FILE}
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully!" | tee -a ${LOG_FILE}
else
    echo "❌ Training failed! Exit code: ${TRAIN_EXIT_CODE}" | tee -a ${LOG_FILE}
    echo "Please check the log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
fi
echo "==========================================" | tee -a ${LOG_FILE}
echo "GSPO Training finished at $(date)" | tee -a ${LOG_FILE}
echo "Exit code: ${TRAIN_EXIT_CODE}" | tee -a ${LOG_FILE}

exit ${TRAIN_EXIT_CODE}
