export CUDA_VISIBLE_DEVICES=2,3,4,5  # 指定使用的GPU编号，如 0 或 1,2
DATA_PATH=/data/jzw/MSRVTT
/home/zxk/miniconda3/envs/creamfl2/bin/torchrun --standalone --nnodes=1 \
  --nproc-per-node=4 \
  --master_port=29487 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=15 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/Videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-2 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.2 \
--use_psa