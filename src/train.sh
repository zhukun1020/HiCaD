

datapath=/users7/kzhu/Datasets/arxiv_survey/data_clean_moref2/title+abs/add_new/original
respath=/users7/kzhu/science_summ/paper2structure/res/data_clean_moref2_title+abs_alldata_large_s1



~/miniconda3/envs/bart_pt10/bin/python run_summarization.py \
    --fp16\
    --num_train_epochs=15\
    --model_name_or_path "allenai/led-large-16384-arxiv" \
    --resume_from_checkpoint /users7/kzhu/science_summ/paper2structure/res/data_clean_moref2_title+abs_alldata_large_s1/checkpoint-2888\
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $datapath/train.json \
    --validation_file $datapath/val.json \
    --test_file $datapath/test.json \
    --text_column source \
    --summary_column target \
    --output_dir $respath \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --learning_rate=4e-5\
    --max_source_length=16384\
    --max_target_length=200\
    --gradient_accumulation_steps=16\
    --label_smoothing_factor 0.1 --weight_decay 0.05 --lr_scheduler_type polynomial \
    --save_strategy epoch --evaluation_strategy no \
    --metric_for_best_model eval_rouge1 \
    --seed 1020

#    --overwrite_output_dir \     --load_best_model_at_end \
