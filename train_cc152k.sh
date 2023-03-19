#train
time=`date '+%Y-%m-%d-%H-%M'`
path='output/'$time
data_path='/data'
vocab_path='/vocab'
gpu=0
module_name=SGR

python3 run.py \
    --id train_cc152k \
    --gpu $gpu \
    --workers 2 \
    --seed 96 \
    --warmup_epoch 10 \
    --warmup_type warmup_sele \
    --data_name cc152k_precomp \
    --noise_ratio 0.4 \
    --warmup_rate 0.3 \
    --p_threshold 0.5 \
    --noise_train noise_soft \
    --noise_tem 0.5 \
    --fit_type bmm \
    --module_name $module_name \
    --output_dir 'output/'$time\
    --data_path $data_path \
    --vocab_path $vocab_path