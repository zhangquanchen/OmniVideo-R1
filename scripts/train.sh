# GSPO https://arxiv.org/pdf/2507.18071
# hyperparameter
# - epsilon = 3e-4 from paper section 5.1
# - epsilon_high = 4e-4 from paper section 5.1
# - steps_per_generation = 4 from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
# - beta = 0: zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
# time_caption_think_answer_format


## opt - mc_acc 
## stage1 -- soft_mc_acc
## opt - think_answer_format
## stage1 - tcta_format
## stage2 - modality_attention
## stage1 - query_intent

# Configure proxy if needed
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port

# Global experiment name variable
EXPERIMENT_NAME="Omnivideo-R1-QI"

# Video sampling parameters to reduce token count
FPS_MAX_FRAMES=64 \
MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=localhost \
MASTER_PORT=29500 \
megatron rlhf \
  --rlhf_type grpo \
  --model /path/to/your/models/Qwen3-Omni-30B-A3B-Instruct \
  --model_type qwen3_omni \
  --save output/gspo_output/${EXPERIMENT_NAME} \
  --add_version false \
  --tensorboard_dir output/gspo_output/${EXPERIMENT_NAME}/runs \
  --wandb_save_dir output/gspo_output/${EXPERIMENT_NAME} \
  --load_safetensors true \
  --save_safetensors true \
  --context_parallel_size 1 \
  --tensor_model_parallel_size 4 \
  --expert_model_parallel_size 4 \
  --pipeline_model_parallel_size 2 \
  --dataset ./data/merged_train_all_qi.jsonl \
  --max_epochs 1 \
  --global_batch_size 16 \
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
  --external_plugins grpo/plugin/omnivideo_r1.py \
  --reward_funcs soft_mc_acc tcta_format query_intent \
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
  --recompute_granularity selective \
  --vit_gradient_checkpointing true \
  --padding_free true \
  --sequence_parallel true \
  --save_interval 50 \
  --no_save_optim true \
  --no_save_rng true \
  --log_interval 1 \
  --num_workers 8 \
  --dataset_num_proc 8 \
  --attention_backend flash \
  --temperature 1.0 \
  --torch_dtype bfloat16 \
  --no_gradient_accumulation_fusion true \
  --system './prompt.txt'
