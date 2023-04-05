# freq - label_len(共8个) 为可调的超参
export CUDA_VISIBLE_DEVICES=1
python -u run.py \
 --task_id sale_test \
 --root_path ./data/train_54_str/ \
 --data_path train_1.csv \
 --freq h \
 --d_model 256 \
 --batch_size 32 \
 --n_heads 8 \
 --modes 64 \
 --moving_avg '7 14 28'\
 --seq_len 30 \
 --label_len 15 \
 --features M \
 --pred_len 15 \
 --e_layers 2 \
 --d_layers 1 \
 --enc_in 34 \
 --dec_in 34 \
 --c_out 34 \
 --itr 1 \
 --train_epochs 1