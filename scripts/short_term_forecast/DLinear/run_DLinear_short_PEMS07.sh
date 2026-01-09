if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_DLinear_short" ]; then
  mkdir ./logs/LongForecasting_DLinear_short
fi

model_name=DLinear
seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0


CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS07.npz \
      --model_id PEMS07_96_12 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 12 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 883 \
      --dec_in 883 \
      --c_out 883 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.01 \
      --itr 1 >logs/LongForecasting_DLinear_short/$model_name'_PEMS07_12'.log

CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS07.npz \
      --model_id PEMS07_96_24 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 24 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 883 \
      --dec_in 883 \
      --c_out 883 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.01 \
      --itr 1 >logs/LongForecasting_DLinear_short/$model_name'_PEMS07_24'.log

CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS07.npz \
      --model_id PEMS07_96_48 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 48 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 883 \
      --dec_in 883 \
      --c_out 883 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.01 \
      --itr 1 >logs/LongForecasting_DLinear_short/$model_name'_PEMS07_48'.log

CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS07.npz \
      --model_id PEMS07_96_96 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 96 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 883 \
      --dec_in 883 \
      --c_out 883 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.01 \
      --itr 1 >logs/LongForecasting_DLinear_short/$model_name'_PEMS07_96'.log



