# from tkinter import scrolledtext
from pyrouge import Rouge155
from pyrouge.utils import log
import logging
import tempfile
from os.path import join
import subprocess as sp
import os
from datasets import load_metric
import nltk
import json

_ROUGE_PATH = '/users7/kzhu/ProgramFiles/Rouge/RELEASE-1.5.5'


def postprocess_text(preds, labels):
    preds = [pred.replace('["',"").replace('"]',"").strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred.replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n"))) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n"))) for label in labels]

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
    metric = load_metric("rouge")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def compute_metrics_s2(decoded_preds, decoded_labels,level_out):
    sumids2 = []
    tmp_sumids = [t.replace(fr"<S{1}>","").strip() for t in decoded_preds]
    for k in range(0,4):
        tmp_sumids = [t.replace(fr"<S{k}>","").strip() for t in tmp_sumids]
    for t in range(len(tmp_sumids)):
        tmp_sum = tmp_sumids[t]
        sum_item = tmp_sum.split("<s")
        tmp_item = list(sum_item)
        # print(sum_item)
        for item in sum_item:
            # print(item)
            if item != "" and item[1] == ">" and int(item[0]) < level_out:
                tmp_item.remove(item)
        sumids2.append("<s".join(tmp_item))
    print(sumids2[0])

    refs = []
    tmp_target = [t.replace(fr"<S{level_out}>","").strip() for t in decoded_labels]
    for k in range(0,4):
        tmp_target = [t.replace(fr"<S{k}>","").strip() for t in tmp_target]
    for t in tmp_target:
        ref_item = t.split("<s")
        tmp_item = list(ref_item)
        for item in ref_item:
            if item != "" and item[1] == ">" and int(item[0]) < level_out:
                tmp_item.remove(item)
        refs.append("<s".join(tmp_item))
    print(refs[0])

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(sumids2, refs)
    metric = load_metric("rouge")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def total_control_onlytitle_huggingface_s2(outpath, ref_res, dec_res,levelout=2):

    with open(ref_res,"r") as f:
        ref_res = f.readlines()

    with open(dec_res,"r") as f:
        dec_res = f.readlines()
     
    score = compute_metrics_s2(dec_res, ref_res, levelout)
    print(score)

    with open(outpath+fr"/ROUGE_s{levelout}.txt",'a',encoding='utf-8') as f:
        f.write("\nonlytitle_huggingface\n"+json.dumps(score)+"\n")

def total_control_onlytitle_huggingface(outpath, ref_res, dec_res):
    # ref_dir = "lcs_ref/"
    # dec_dir = "lcs_dec/"

    # ref_res = "lcs.ref"
    # dec_res = "lcs.dec"

    # with open(ref_res,"r") as f:
    #     ref_res = f.readlines()

    # with open(dec_res,"r") as f:
    #     dec_res = f.readlines()
     
    score = compute_metrics(dec_res, ref_res)
    print(score)

    with open(outpath+"/ROUGE.txt",'a',encoding='utf-8') as f:
        f.write("onlytitle_huggingface\n"+json.dumps(score)+"\n\n")

    # model = dec_dir
    # print(model)
    # with open('result_bart_500.txt',"a") as f3:
    #     f3.write(model+"\t")
    #     f3.write("\t".join(score))
    #     f3.write("\n")


def eval_rouge(dec_dir, ref_dir, Print=True):
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    cmd = '-c 95 -r 1000 -n 2 -a -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
                dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
                ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
                join(tmp_dir, 'dec'), dec_pattern,
                join(tmp_dir, 'ref'), ref_pattern,
                join(tmp_dir, 'settings.xml'), system_id=1
            )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
                + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
                + cmd
                + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
        R_1 = float(output.split('\n')[3].split(' ')[3])
        R_2 = float(output.split('\n')[7].split(' ')[3])
        R_L = float(output.split('\n')[11].split(' ')[3])
        print(output)
    if Print is True:
        # rouge_path = join(dec_dir, '../'+save_name+'-ROUGE.txt')
        rouge_path = join(dec_dir, '../'+'ROUGE.txt')
        with open(rouge_path, 'a') as f:
            print(output, file=f)
    return [str(R_1), str(R_2), str(R_L)]
	

def postprocess_text_test(preds):
    preds = [pred.replace("<q>","\n").strip() for pred in preds]
    # labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred.replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n").replace("<eos>",""))) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label.replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n"))) for label in labels]

    return preds

def total_control(ref_res, dec_res, ref_dir, dec_dir, save_name):
    # ref_dir = "lcs_ref/"
    # dec_dir = "lcs_dec/"

    # ref_res = "lcs.ref"
    # dec_res = "lcs.dec"

    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(dec_dir):
        os.mkdir(dec_dir)

    with open(ref_res,"r") as f:
        data = f.readlines()
    refs = postprocess_text_test(data)   
    for count in range(len(refs)):
        with open(ref_dir +str(count)+".ref","w") as f2:
            f2.write(refs[count])

    with open(dec_res,"r") as f:
        data = f.readlines()
    preds = postprocess_text_test(data)  
    for count in range(len(preds)):
        with open(dec_dir+str(count)+".dec","w") as f2:
            f2.write(preds[count])
            # f2.write(dec_res[count].replace("\t","\n"))

        
    score = eval_rouge(dec_dir, ref_dir, Print=True)
    
    # model = dec_dir
    # print(model)
    # with open('result_bart_500.txt',"a") as f3:
    #     f3.write(model+"\t")
    #     f3.write("\t".join(score))
    #     f3.write("\n")


def total_control_onlytitle(ref_res, dec_res, ref_dir, dec_dir):
    # ref_dir = "lcs_ref/"
    # dec_dir = "lcs_dec/"

    # ref_res = "lcs.ref"
    # dec_res = "lcs.dec"

    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(dec_dir):
        os.mkdir(dec_dir)

    # with open(ref_res,"r") as f:
    #     ref_res = f.readlines()
    for count in range(len(ref_res)):
        with open(ref_dir +str(count)+".ref","w") as f2:
            f2.write(ref_res[count].replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n").strip())

    # with open(dec_res,"r") as f:
    #     dec_res = f.readlines()
    for count in range(len(dec_res)):
        with open(dec_dir+str(count)+".dec","w") as f2:
            f2.write(dec_res[count].replace("<s1>","\n").replace("<s2>","\n").replace("<s3>","\n").strip())
        
    score = eval_rouge(dec_dir, ref_dir, Print=True)

    model = dec_dir
    print(model)
    with open('result_bart_500.txt',"a") as f3:
        f3.write(model+"/onlytitle_rouge155\t")
        f3.write("\t".join(score))
        # f3.write("\n")

if __name__ == '__main__':
    # ref_dir = "/users7/kzhu/science_summ/paper2structure/test_rouge/ref/"
    # dec_dir = "/users7/kzhu/science_summ/paper2structure/test_rouge/dec/"

    # ref_res = "/users7/kzhu/science_summ/paper2structure/res_predict/led_base_BigSurvey_16384_data9_3nopattern_3/checkpoint-21000-/block3/refs.txt"
    # dec_res = "/users7/kzhu/science_summ/paper2structure/res_predict/led_base_BigSurvey_16384_data9_3nopattern_3/checkpoint-21000-/block3/generated_predictions.txt"
    # save_name = 'test'

    # total_control(ref_res, dec_res, ref_dir, dec_dir, save_name)

    # total_path = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data3"
    # for dir in os.listdir(total_path):
    #     path = os.path.join(total_path,dir,"block_5")
    #     if os.path.isdir(path):
    #         ref_dir = os.path.join(path,"ref/")
    #         dec_dir = os.path.join(path,"dec/")
    #         ref_res = os.path.join(path,"refs.txt")
    #         dec_res = os.path.join(path,"generated_predictions.txt")
    #         save_name = 'test_onlytitle'
    #         total_control_onlytitle(ref_res, dec_res, ref_dir, dec_dir, save_name)

    path = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data3/checkpoint-30000/block_4"
    path = "/users7/kzhu/science_summ/paper2structure/res/data_clean_moref2_title+abs_alldata_s1/"
    ref_res = os.path.join(path,"generated_predictions.txt")
    dec_res = os.path.join(path,"generated_predictions_ref.txt")
    total_control_onlytitle_huggingface_s2(path, ref_res, dec_res,levelout=0)


    