if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_iTransformer_short" ]; then
  mkdir ./logs/LongForecasting_iTransformer_short
fi

model_name=iTransformer
seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS03.npz \
      --model_id PEMS03_96_12 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 12 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 358 \
      --dec_in 358 \
      --c_out 358 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_iTransformer_short/$model_name'_PEMS03_12'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS03.npz \
      --model_id PEMS03_96_24 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 24 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 358 \
      --dec_in 358 \
      --c_out 358 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_iTransformer_short/$model_name'_PEMS03_24'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS03.npz \
      --model_id PEMS03_96_48 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 48 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 358 \
      --dec_in 358 \
      --c_out 358 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_iTransformer_short/$model_name'_PEMS03_48'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS03.npz \
      --model_id PEMS03_96_96 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 96 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 358 \
      --dec_in 358 \
      --c_out 358 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.001 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_iTransformer_short/$model_name'_PEMS03_96'.log