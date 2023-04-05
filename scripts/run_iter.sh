# iter over storeid

# use GPU
export CUDA_VISIBLE_DEVICES=0

for storeid in {1:54}
do

python -u run.py \
 --task_id store$storeid \
 --root_path ./data/train_54_str/ \
 --data_path train_$storeid.csv \
 --freq h \
 --d_model 512 \
 --batch_size 32 \
 --n_heads 8 \
 --modes 30 \
 --moving_avg '3 7 14 28'\
 --seq_len 30 \
 --label_len 15 \
 --features M \
 --pred_len 16 \
 --e_layers 2 \
 --d_layers 1 \
 --enc_in 34 \
 --dec_in 34 \
 --c_out 34 \
 --itr 1 \
 --train_epochs 10

 done