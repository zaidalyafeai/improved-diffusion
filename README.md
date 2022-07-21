# Text-writing denoising diffusion

This is a heavily modified fork of OpenAI's [improved-diffusion](https://github.com/openai/improved-diffusion) codebase, adding many features I've found useful or interesting:

- Support for _transcription conditioning_: training a model to write arbitrary text into an image in a contextually appropriate manner
- Support for caption conditioning, a la Imagen/GLIDE/etc., using a variety of approaches
- Classifier-free guidance
  - (I forked this repo instead of OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) at the beginning of my diffusion research, but I ended up adding guidance later)
- [Noise conditioning](https://cascaded-diffusion.github.io/assets/cascaded_diffusion.pdf) for upsamplers, which dramatically improves upsampling quality in my experience
- The dynamic thresholding trick from the [Imagen](https://imagen.research.google/paper.pdf) paper
- Higher-order numerical methods for accelerated DDIM sampling (PRK and PLMS from [this paper](https://openreview.net/forum?id=PlKWVd2yBkY))
- Arithmetic averaging before exponential averaging in training, greatly accelerating EMA convergence ([see my post here](https://nostalgebraist.tumblr.com/post/675308518749978624/exponential-moving-averages-emas-are))
- Antialiasing when downsampling ground-truth images for training upsamplers, which (also) dramatically improves upsampling quality in my experience

**This is a personal research codebase.**  Code style and quality varies.  The code was written quickly, for personal use, to prototype and evaluate ideas I didn't necessarily expect to use in the long term.

The section below on usage contains some pointers for how to use the code in the same way I do, but should not be considered a comprehensive guide.

## External resources

- [Colab demo](https://colab.research.google.com/drive/1XYGfJr-BTDRcHVl0K5i4-nWv9d9RL3SY?usp=sharing) of a fully trained cascaded diffusion stack (recommended!)
- [Blog post](https://nostalgebraist.tumblr.com/post/672300992964050944/franks-image-generation-model-explained) from Jan 2022 detailing my research on transcription conditioning

## Architectural details

For _transcription conditioning_, I use

- a character-level transformer encoder using T5-style relative position embeddings, trained end-to-end with the model, and 
- layers of cross-attention added to the U-net which only query over the outputs of the encoder not a concatenation of image + encoder outputs), using axial learned position embeddings
 
An optional variant I call "weave attention" adds an intermediate layer of image-to-text attention, followed a transformer-style MLP, before the text-to-image attention layer.


For _caption conditioning_, I use a pretrained CLIP text encoder.  I support several ways of connecting it to the image model, including 

1. adding the CLIP encoder's activation at the final position to the main U-net embedding stream (as in GLIDE)
2. allowing its existing attention layers to attend additionally to the final or penultimate layer activations of the CLIP encoder (as in both GLIDE and Imagen)

In my experience, "Imagen-style" configuration (only use option 2 above, with penultimate activations) works best.

## Usage

For basic usage, refer to the README of the [parent repo](https://github.com/openai/improved-diffusion).

Quick review of some terminology I throughout the code:

- `txt`: transcription conditioning
- `capt`, `capts`: description conditioning
- `safebox`: rectangle enclosing the bounding boxes of all text in a training image. Used with cropping augmentation to prevent crops from cutting off text.
- `es`: short for "empty string," refers to training augmentations that work differently for images which contain no text
- `noise_cond_ts`: diffusion timestamp for noise conditioning, from a 1000-step cosine-schedule diffusion process

The following block of python code will train something very close to my 64x64 base model.

- To provide ground-truth transcriptions, include `.txt` files with the same names as the training images, in the same directories.
- Captions, and other inputs like "safeboxes," are provided in json files that use a special key syntax to refer to locations of training images on disk.  TODO: document this.

```python
LOGDIR=""  # fill in directory to save checkpoints/etc.

MODEL_FLAGS=""
TRAIN_FLAGS=""

## configuration of the u-net
MODEL_FLAGS += " --image_size 64 --num_res_blocks 2"
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
TRAIN_FLAGS+= " --use_special_crop_for_empty_string True"
TRAIN_FLAGS+= " --use_random_safebox_for_empty_string True"
TRAIN_FLAGS+= " --crop_prob_es 0.95 --crop_prob 0.95"
TRAIN_FLAGS+= " --crop_min_scale 0.44445"
TRAIN_FLAGS+= " --safebox_path safeboxes.jsonl"

# original sizes of the training images - prevents cropping from zooming in more than 1:1 
TRAIN_FLAGS+= " --px_scales_path px_scales_path.jsonl"

# flip left/right augmentation probability, only used the image has no text
TRAIN_FLAGS+= " --flip_lr_prob_es 0.35"

MODEL_FLAGS+=" --using_capt 1"  # use description conditioning

## configuration for description conditioning

# use frozen pretrained CLIP ViT-L/14@336px text encoder
MODEL_FLAGS+=" --clipname ViT-L/14@336px --freeze_capt_encoder 1"
# imagen-style attention approach
MODEL_FLAGS+=" --glide_style_capt_attn 1 --glide_style_capt_emb 0 --clip_use_penultimate_layer 1"

TRAIN_FLAGS+= " --capt_path capts.json"  # json file mapping image paths to captions

TRAIN_FLAGS+= " --use_fp16 true --use_amp True"  # use torch AMP rather than OpenAI's hand-built AMP

# drop rate for conditioning to support guidance
TRAIN_FLAGS+=" --txt_pdrop 0.1"
TRAIN_FLAGS+=" --capt_pdrop 0.1 --all_pdrop 0.1"

## model averaging
TRAIN_FLAGS+= " --ema_rate 0.9999"

# do an arithmetic average from this step until step 1/(1-ema_rate) - dramatically accelerates EMA convergence
TRAIN_FLAGS+= " --arithmetic_avg_from_step 0"
# increment this manually every time you start the training anew from a checkpoint :(
TRAIN_FLAGS+= " --arithmetic_avg_extra_shift 0"

## performance-related flags - adapt these if you run out of memory / etc
TRAIN_FLAGS+=" --microbatch 36"
TRAIN_FLAGS+= " --perf_no_ddl 1"
TRAIN_FLAGS+= " --perf_pin_memory 1"
TRAIN_FLAGS+= " --perf_prefetch_factor 4"
TRAIN_FLAGS+= " --perf_num_workers 8"
TRAIN_FLAGS+= " --silu_impl fused"
TRAIN_FLAGS+= " --cudnn_benchmark 1"

# saves the first batch of inputs in human-readable form, useful for debuggning
TRAIN_FLAGS+=" --save_first_batch 1"

## learning rate, etc.
TRAIN_FLAGS+=" --lr 1e-4 --batch_size 504 --lr_warmup_steps 200"

TRAIN_FLAGS+=" --fp16_scale_growth 2e-3"
TRAIN_FLAGS+=f" --config_path {LOGDIR}config.json"

TRAIN_FLAGS += " --log_interval 10 --verbose 0"
TRAIN_FLAGS+=" --save_interval 2000 --autosave 0"  # todo: support gcs autosave for arbitrary buckets

RESUME_FLAGS = ""  # if training from scratch

!TOKENIZERS_PARALLELISM=false OPENAI_LOGDIR={LOGDIR} python3 scripts/image_train.py \
--data_dir path_to_data/ {MODEL_FLAGS} {DIFFUSION_FLAGS} {TRAIN_FLAGS} {RESUME_FLAGS}
```

Training an upsampler is similar, but I recommend passing additional arguments like

```python
# use noise conditioning
TRAIN_FLAGS+=" --noise_cond 1"
# only sample conditioning noise from 0-600 rather than 0-1000
TRAIN_FLAGS+=" --noise_cond_max_step 600"  

# use antialiasing when downsampling ground-truth images
#this dramatically improves upsampling quality in my experience
TRAIN_FLAGS+=" --antialias true"

# use bicubic (rather than bilinear) resampling when downsampling ground-truth images
TRAIN_FLAGS+= --bicubic_down true"
```
