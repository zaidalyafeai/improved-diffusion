BATCH_SIZE = "128"
IMG_SIZE = "64"
LOGDIR = f"calliargen/model_10k_mulfont_bs_{BATCH_SIZE}_{IMG_SIZE}x{IMG_SIZE}_with_t5/"  # fill in directory to save checkpoints/etc.
DATADIR = "CalliarGen/data_10k_mulfont_64x64_en"
MODEL_FLAGS=""
TRAIN_FLAGS=""

## configuration of the u-net
MODEL_FLAGS += f" --image_size {IMG_SIZE} --num_res_blocks 2"
# num_heads controls attention in middle_block, channels_per_head is used elsewhere
MODEL_FLAGS += " --num_channels 256 --channels_per_head 64 --num_heads 16"
MODEL_FLAGS += " --channel_mult 1,2,2,4,4"
MODEL_FLAGS += " --learn_sigma True --attention_resolutions 4,8,16"
MODEL_FLAGS += " --use_scale_shift_norm True --resblock_updown True"
MODEL_FLAGS += " --monochrome 0 --monochrome_adapter 0"

## configuration of the diffusion process
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
DIFFUSION_FLAGS+=" --rescale_learned_sigmas False --rescale_timesteps False"

MODEL_FLAGS += " --txt 1" # use transcription conditioning

## configuration of the transcription encoder
MODEL_FLAGS += " --txt_dim 512 --txt_depth 4 --max_seq_len 384"
MODEL_FLAGS += " --txt_rezero 0 --txt_ff_glu 1 --txt_ff_mult 3"
MODEL_FLAGS += " --txt_t5 True"
MODEL_FLAGS += " --char_level True"

## configuration of the transcription-to-image cross-attention
MODEL_FLAGS += " --weave_attn True --weave_qkv_dim_always_text True"
MODEL_FLAGS+= " --weave_use_ff_gain 1"
MODEL_FLAGS += " --weave_ff_rezero False --weave_ff_glu False --weave_ff_mult 2"
MODEL_FLAGS += " --cross_attn_use_layerscale 0 --cross_attn_init_gain 1 --cross_attn_gain_scale 1"
MODEL_FLAGS += " --cross_attn_rezero 0 --cross_attn_rezero_keeps_prenorm 1"
MODEL_FLAGS += " --txt_avoid_groupnorm 0 --cross_attn_q_t_emb 1 --cross_attn_orth_init 1 --txt_attn_before_attn 0"
MODEL_FLAGS += "  --cross_attn_channels_per_head 128 --txt_resolutions 8,16,32 --txt_output_layers_only 1"

## configuration for image augmentations during training
TRAIN_FLAGS+= " --use_special_crop_for_empty_string False"
TRAIN_FLAGS+= " --use_random_safebox_for_empty_string False"
TRAIN_FLAGS+= " --crop_prob_es 0 --crop_prob 0"
TRAIN_FLAGS+= " --crop_min_scale 0.44445"
TRAIN_FLAGS+= " --safebox_path safeboxes.jsonl"

# original sizes of the training images - prevents cropping from zooming in more than 1:1 
TRAIN_FLAGS+= " --px_scales_path px_scales_path.jsonl"

# flip left/right augmentation probability, only used the image has no text
TRAIN_FLAGS+= " --flip_lr_prob_es 0.0"

MODEL_FLAGS+=" --using_capt 1"  # use description conditioning

# # configuration for description conditioning

# use frozen pretrained CLIP ViT-L/14@336px text encoder
# MODEL_FLAGS+=" --clipname ViT-L/14@336px --freeze_capt_encoder 1"
MODEL_FLAGS+=" --clipname t5-v1_1-xxl --freeze_capt_encoder 1"
# imagen-style attention approach
MODEL_FLAGS+=" --glide_style_capt_attn 1 --glide_style_capt_emb 0 --clip_use_penultimate_layer 1"

TRAIN_FLAGS+= f" --capt_path {DATADIR}/capts.json"  # json file mapping image paths to captions

TRAIN_FLAGS+= " --use_fp16 true --use_amp True"  # use torch AMP rather than OpenAI's hand-built AMP

# drop rate for conditioning to support guidance
TRAIN_FLAGS+=" --txt_pdrop 0.0"
TRAIN_FLAGS+=" --capt_pdrop 0.1 --all_pdrop 0.0"

## model averaging
TRAIN_FLAGS+= " --ema_rate 0.9999"

# do an arithmetic average from this step until step 1/(1-ema_rate) - dramatically accelerates EMA convergence
TRAIN_FLAGS+= " --arithmetic_avg_from_step 0"
# increment this manually every time you start the training anew from a checkpoint :(
TRAIN_FLAGS+= " --arithmetic_avg_extra_shift 0"

## performance-related flags - adapt these if you run out of memory / etc
TRAIN_FLAGS+=" --microbatch 16" # batch size is split into microbatches
TRAIN_FLAGS+= " --perf_no_ddl 1"
TRAIN_FLAGS+= " --perf_pin_memory 1"
TRAIN_FLAGS+= " --perf_prefetch_factor 4"
TRAIN_FLAGS+= " --perf_num_workers 8"
TRAIN_FLAGS+= " --silu_impl fused"
TRAIN_FLAGS+= " --cudnn_benchmark 1"

# saves the first batch of inputs in human-readable form, useful for debuggning
TRAIN_FLAGS+=" --save_first_batch 1"

## learning rate, etc.
TRAIN_FLAGS+=f" --lr 1e-4 --batch_size {BATCH_SIZE} --lr_warmup_steps 200"

TRAIN_FLAGS+=" --fp16_scale_growth 2e-3"
TRAIN_FLAGS+=f" --config_path {LOGDIR}config.json"

TRAIN_FLAGS+= " --log_interval 10 --verbose 0"
TRAIN_FLAGS+=" --save_interval 5000 --autosave 0"  # todo: support gcs autosave for arbitrary buckets
TRAIN_FLAGS+=" --use_wandb True"
# RESUME_FLAGS = "--resume_checkpoint calliargen/model_10k_mulfont_bs_128_64x64_with_t5/model005000.pt"  # if training from scratch
RESUME_FLAGS = ""
import os

os.system(f"TOKENIZERS_PARALLELISM=false OPENAI_LOGDIR={LOGDIR} python3 scripts/image_train.py --data_dir {DATADIR}/train {MODEL_FLAGS} {DIFFUSION_FLAGS} {TRAIN_FLAGS} {RESUME_FLAGS}")
