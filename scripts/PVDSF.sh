if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_names=("PVDSF") 
data_path_names=("宁夏永宁数据_5min_天气.csv" "宁夏中宁数据_5min_天气.csv" "宁夏中卫数据_5min_天气.csv" "山东枣庄数据_5min_天气.csv")
random_seed=2021
variable_num=6
enc_in=$((variable_num + 1))

root_path_name=./dataset/
data_name=custom

for model_name in "${model_names[@]}"
do
    for data_path_name in "${data_path_names[@]}"
    do
        if [ ! -d "./logs/$model_name" ]; then
            mkdir ./logs/$model_name
        fi

        if [ ! -d "./logs/$model_name/$data_path_name" ]; then
            mkdir ./logs/$model_name/$data_path_name
        fi

        for seq_len in 12 72 288  # 12 72 288 
        do
            for e_layers in 3 
            do
                for d_model in 128 
                do
                    for learning_rate in 0.0001 
                    do
                        for patch_len in 6 
                        do
                            for down_sampling_window in 2  
                            do
                                for down_sampling_layers in 2  
                                do
                                    for pred_len in 12  
                                    do
                                        seq_len=$seq_len
                                        pred_len=$seq_len

                                        python3 -u run_longExp.py \
                                        --d_layers 3 \
                                        --random_seed $random_seed \
                                        --is_training 1 \
                                        --root_path $root_path_name \
                                        --data_path $data_path_name \
                                        --model_id "${model_name}_${data_path_name}_${seq_len}_${pred_len}" \
                                        --model $model_name \
                                        --data $data_name \
                                        --features MS \
                                        --seq_len $seq_len \
                                        --pred_len $pred_len \
                                        --label_len 6 \
                                        --enc_in $enc_in \
                                        --dec_in $enc_in \
                                        --c_out $enc_in \
                                        --e_layers $e_layers \
                                        --n_heads 8 \
                                        --d_model $d_model \
                                        --d_ff 256 \
                                        --dropout 0.2 \
                                        --fc_dropout 0.2 \
                                        --head_dropout 0 \
                                        --patch_len $patch_len \
                                        --des 'Exp' \
                                        --train_epochs 100 \
                                        --patience 20 \
                                        --target '每分钟发电量 (kWh)' \
                                        --itr 1 \
                                        --batch_size 64 \
                                        --down_sampling_window $down_sampling_window \
                                        --down_sampling_layers $down_sampling_layers \
                                        --learning_rate $learning_rate >logs/$model_name/$data_path_name/${model_name}_${data_path_name}_${seq_len}_${pred_len}_${e_layers}_${d_model}_${learning_rate}_mape.log # ${model_name}_${data_path_name}_${seq_len}_${pred_len}_${e_layers}_${d_model}_${learning_rate}_${patch_len}_${variable_num}_${stride}
                                    done
                                done
                            done
                        done
                    done
                done
            done 
        done
    done
done
