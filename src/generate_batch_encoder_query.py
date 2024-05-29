# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 7:38 下午
# @Author  : Xiachong Feng
# @File    : generate_batch.py
# @Software: PyCharm

import argparse
import codecs
import json
from tqdm import tqdm
import os
import re

# from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import LEDTokenizer
from transformers import LEDForConditionalGeneration
from test_rouge import total_control_onlytitle, total_control_onlytitle_huggingface
from datasets import load_metric
import numpy as np
import nltk
from test_bleu import compute_metrics_title, compute_metrics_label
from test_distance import compute_TEDS

'''

cd /users7/kzhu/science_summ/paper2structure/
python generate_batch.py\
    --model /users7/kzhu/science_summ/paper2structure/res/led_base_BigSurvey_16384_data9_3nopattern_3/checkpoint-36000 \
    --output_dir /users7/kzhu/science_summ/paper2structure/res_predict/led_base_BigSurvey_16384_data9_3nopattern_3/checkpoint-30000-block4 \
    --no_repeat_ngram_size 4\
    --batch 4\
    --repetition_penalty 1.2\
    --max_source_length 16384\
    --data_path /users7/kzhu/Datasets/BigSurvey/Survey_paper/abstract/final_data9_3nopattern_3/test.json\
    --generation_max_length 350

python generate_batch.py\
    --model  /users7/kzhu/science_summ/paper2structure/res/led_BigSurvey_16384_data5_simorder/checkpoint-30000\
    --output_dir /users7/kzhu/science_summ /paper2structure/res/led_BigSurvey_16384_data5_simorder/checkpoint-30000/block_4_model5_data3 \
    --no_repeat_ngram_size 4\
    --batch 1\
    --repetition_penalty 1.2\
    --max_source_length 4096\
    --data_path /users7/kzhu/Datasets/BigSurvey/Survey_paper/abstract/final_data3/test.json\
    --generation_max_length 1024   
    
'''

       
metric = load_metric("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(decoded_preds, decoded_labels):
    # preds, labels = eval_preds
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def load_json(args):
    with codecs.open(args.data_path, "r", "utf-8") as f:
        datas = []
        for line in f:
            datas.append(json.loads(line))
    return datas



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, default=None)
    # parser.add_argument("--valid_or_test", type=str, default='test')
    # parser.add_argument("--model_type", type=str, default='CPT')
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--generation_max_length", type=int, default=None)
    parser.add_argument("--summary_column", type=str, default=None)
    parser.add_argument("--queryadd_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)


    args = parser.parse_args()
    
    # ck_name = args.model.split("/")[-1]
    # if len(ck_name.split("-"))<3:
    #     exit(0)
    args.output_dir = args.model.replace("res","res_predict")

    # assert not os.path.exists(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if os.path.exists(args.output_dir + fr"-block{args.no_repeat_ngram_size}"):
    #     exit(0)
    print(args.output_dir)
    args.output_dir = args.output_dir + fr"/block{args.no_repeat_ngram_size}"

    # assert not os.path.exists(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)  
    else:
        exit(0)
    
    

    # model = CPTForConditionalGeneration.from_pretrained(args.model)
    # model.to("cuda")
    tokenizer = LEDTokenizer.from_pretrained(args.model)  
    model = LEDForConditionalGeneration.from_pretrained(args.model)

    # tokenizer.bos_token = tokenizer.convert_ids_to_tokens(model.config.decoder_start_token_id)
    # model.config.forced_eos_token_id = model.config.eos_token_id = 10
    # tokenizer.eos_token = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)  
    # model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    datas = load_json(args)

    if len(datas) % args.batch == 0:
        batch_num = (len(datas) // args.batch)
    else:
        batch_num = (len(datas) // args.batch) + 1
    texts = []
    # tmp_targets = []
    targets = []
    querys = []    
    level_input = 0
    level_out = 2    
    for data in datas:
        texts.append(data["source"])
        targets.append(data[args.summary_column])
        if args.queryadd_column:
            querys.append(fr"{data['query']} {data[args.queryadd_column]}")
    

    targets = ["\n".join(nltk.sent_tokenize(pred.replace("<s>","").replace("</s>",""))) for pred in targets]

    summaries = []
    refs = []
    for batch_id in tqdm(range(batch_num)):
        # print(fr"------------------{batch_id}---------------")
        tmp_input = texts[batch_id * args.batch:(batch_id + 1) * args.batch]

        tmp_title = querys[batch_id * args.batch:(batch_id + 1) * args.batch]

        tmp_sumids = []
        for k in range(level_input,level_input+1):#level_out

            tmp_input = [fr"{tmp_title[i]}"+tmp_input[i] for i in range(len(tmp_input))]                

            tmp_transform_inputs = tokenizer(tmp_input, max_length=args.max_source_length, padding=True, truncation=True,
                                            add_special_tokens=True, return_tensors='pt')
            tmp_transform_inputs.to("cuda")
            tmp_summary_ids = model.generate(tmp_transform_inputs['input_ids'], min_length=10,
                                            max_length=args.generation_max_length, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                                            length_penalty=1.0, 
                                            num_beams=4,
                                            use_cache=True,
                                            # output_attentions=True,
                                            # output_hidden_states=True,
                                            # return_dict=True,
                                            # top_p=0.8,
                                            # top_k=10,
                                            early_stopping=False)
            # print("---------------------------------------------")

            tmp_sumids = tokenizer.batch_decode(tmp_summary_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

        summaries.extend(tmp_sumids)
        

    output_prediction_file = os.path.join(args.output_dir, "generated_predictions_all.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(summaries))  

    output_ref_file = os.path.join(args.output_dir, "refs_all.txt")
    with open(output_ref_file, "w") as writer:
        writer.write("\n".join(targets)) 

    
    ref_dir = os.path.join(args.output_dir, "lcs_ref/")
    dec_dir = os.path.join(args.output_dir, "lcs_dec/")
    total_control_onlytitle(targets, summaries, ref_dir, dec_dir)
    total_control_onlytitle_huggingface(args.output_dir, targets, summaries)

    # compute metrics for different level headings
    for m in range(1,4):
        sumids = []
        tmp_sumids = [t.replace(fr"<S{level_out}>","").strip() for t in summaries]
        for k in range(0,4):
            tmp_sumids = [t.replace(fr"<S{k}>","").strip() for t in tmp_sumids]
        for t in tmp_sumids:
            sum_item = t.split("<s")
            tmp_item = list(sum_item)
            # print(sum_item)
            for item in sum_item:
                # print(item)
                if item != "" and item[1] == ">" and int(item[0]) != m:
                    tmp_item.remove(item)
            sumids.append("<s".join(tmp_item))
        refs = []
        tmp_target = [t.replace(fr"<S{level_out}>","").strip() for t in targets]
        for k in range(0,4):
            tmp_target = [t.replace(fr"<S{k}>","").strip() for t in tmp_target]
        for t in tmp_target:
            ref_item = t.split("<s")
            tmp_item = list(ref_item)
            for item in ref_item:
                if item != "" and item[1] == ">" and int(item[0]) != m:
                    tmp_item.remove(item)
            refs.append("<s".join(tmp_item))

        output_prediction_file = os.path.join(args.output_dir, fr"generated_predictions_s{m}.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(sumids))

        output_ref_file = os.path.join(args.output_dir, fr"refs_s{m}.txt")
        with open(output_ref_file, "w") as writer:
            writer.write("\n".join(refs))  

        total_control_onlytitle(refs, sumids, ref_dir, dec_dir)

        total_control_onlytitle_huggingface(args.output_dir, refs, sumids)


    compute_metrics_label(args.output_dir, summaries, targets)
    compute_metrics_title(args.output_dir, summaries, targets)
    compute_TEDS(args.output_dir, summaries=summaries, refs=targets)
    

