if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_TimesNet_short" ]; then
  mkdir ./logs/LongForecasting_TimesNet_short
fi


model_name=TimesNet
seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0


CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS04.npz \
      --model_id PEMS04_96_12 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 12 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 307 \
      --dec_in 307 \
      --c_out 307 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 6 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_TimesNet_short/$model_name'_PEMS04_12'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS04.npz \
      --model_id PEMS04_96_24 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 24 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 307 \
      --dec_in 307 \
      --c_out 307 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 24 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_TimesNet_short/$model_name'_PEMS04_24'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS04.npz \
      --model_id PEMS04_96_48 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 48 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 307 \
      --dec_in 307 \
      --c_out 307 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 48 \
      --station_lr 0.0005 \
      --learning_rate 0.0001 \
      --itr 1 >logs/LongForecasting_TimesNet_short/$model_name'_PEMS04_48'.log

CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_CDN \
      --root_path ./datasets/PEMS/ \
      --data_path PEMS04.npz \
      --model_id PEMS04_96_96 \
      --model $model_name \
      --data PEMS \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len 96 \
      --e_layers 4 \
      --d_layers 2 \
      --factor 3 \
      --enc_in 307 \
      --dec_in 307 \
      --c_out 307 \
      --des 'Exp' \
      --station_type $station_type \
      --top_k 3 \
      --max_period 24 \
      --station_lr 0.0005 \
      --learning_rate 0.0005 \
      --itr 1 >logs/LongForecasting_TimesNet_short/$model_name'_PEMS04_96'.log

