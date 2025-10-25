export CUDA_VISIBLE_DEVICES=2,3,4,5  # 指定使用的GPU编号，如 0 或 1,2
DATA_PATH=/data/jzw/MSRVTT
# python -m torch.distributed.launch --nproc_per_node=4 \
/home/zxk/miniconda3/envs/creamfl2/bin/torchrun --standalone --nnodes=1 \
  --nproc-per-node=4 \
  --master_port=29486 \
main_task_retrieval.py  --do_eval \
  --num_thread_reader=0 \
  --batch_size_val 128 \
  --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
  --features_path ${DATA_PATH}/Videos \
  --output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
  --datatype msrvtt \
  --feature_framerate 1 \
  --max_words 32 \
  --max_frames 12 \
  --slice_framepos 2 \
  --loose_type \
  --linear_patch 2d \
  --sim_header meanP \
  --pretrained_clip_name "ViT-B/32" \
  --init_model ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.2 \
  --eval_with_slots \
  --use_psa