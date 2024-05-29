
path0="/users7/kzhu/science_summ/paper2structure/res2/cs_large_data2_s3"
max_source_length=16384
batch_size=1
no_repeat_ngram_size=3

data_path="/users7/kzhu/Datasets/arxiv/23.04.23/combine_data/cs_all2_split/test.json"

generation_max_length=500

for dir in $path0/*  
do
    if test -d $dir;
    then 
        echo $dir 
        ~/miniconda3/envs/bart_pt10/bin/python generate_batch_encoder_query.py\
             --model $dir \
             --output_dir $dir/block_3\
             --no_repeat_ngram_size 3\
             --text_column source \
             --summary_column target \
             --batch $batch_size\
             --max_source_length $max_source_length\
             --data_path $data_path\
             --generation_max_length $generation_max_length
    fi
done
