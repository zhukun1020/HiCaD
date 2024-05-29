from datasets import load_metric
import os
import nltk
import json
from bleu import Bleu
import re

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    # preds = ["\t<".join(pred.split(" <")).split("\t") for pred in preds]
    # labels = ["\t<".join(label.split(" <")).split("\t") for label in labels]
    preds = [nltk.word_tokenize(pred.replace("<","").replace(">","")) for pred in preds]
    labels = [[nltk.word_tokenize(label.replace("<","").replace(">",""))] for label in labels] 
    
    # preds = [nltk.word_tokenize(pred.replace("<s1>","").replace("<s2>","").replace("<s3>","")) for pred in preds]
    # labels = [[nltk.word_tokenize(label.replace("<s1>","").replace("<s2>","").replace("<s3>",""))] for label in labels]   

    # pattern = re.compile("<s[1-3]>")
    # for pred in preds:
    #     res_list = re.findall(pattern,pred)
    #     print(res_list)
    #     for res in res_list:

    # preds = [re.findall(pattern,pred) for pred in preds]
    # labels = [[re.findall(pattern,label)] for label in labels]   

    return preds, labels

def postprocess_text_label(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    #structure
    pattern = re.compile("<s[1-3]>")
    preds = [re.findall(pattern,pred) for pred in preds]
    labels = [[re.findall(pattern,label)] for label in labels]   

    return preds, labels

def postprocess_text_title(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    #title
    preds = [nltk.word_tokenize(pred.replace("<s1>","").replace("<s2>","").replace("<s3>","")) for pred in preds]
    labels = [[nltk.word_tokenize(label.replace("<s1>","").replace("<s2>","").replace("<s3>",""))] for label in labels]   
    # preds = [pred for pred in preds]
    # labels = [[label] for label in labels]  

    return preds, labels

def compute_metrics_label(output_dir, decoded_preds, decoded_labels):
    # preds, labels = eval_preds
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text_label(decoded_preds, decoded_labels)
    # print(str(decoded_labels[0]))
    metric = Bleu()
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}

    output_bleu_file = os.path.join(output_dir, "BLEU.txt")
    with open(output_bleu_file, "a") as writer:
        writer.write("label\n")
        writer.write(json.dumps(result,indent=4)+"\n")
        # json.dump(result,writer,indent=4)
    with open('result_bleu_500.txt',"a") as f3:
        f3.write(output_dir+"/label\t")
        f3.write(json.dumps(result))
        f3.write("\n")

    return result

def compute_metrics_title(output_dir, decoded_preds, decoded_labels):

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text_title(decoded_preds, decoded_labels)
    # print(str(decoded_labels[0]))
    metric = Bleu()
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    output_bleu_file = os.path.join(output_dir, "BLEU.txt")
    with open(output_bleu_file, "a") as writer:
        writer.write("title\n")
        writer.write(json.dumps(result,indent=4)+"\n")
        # json.dump(result,writer,indent=4)
    with open('result_bleu_500.txt',"a") as f3:
        f3.write(output_dir+"/title\t")
        f3.write(json.dumps(result))
        f3.write("\n")

    return result   

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
    decoded_preds, decoded_labels = postprocess_text_split(decoded_preds, decoded_labels)
    # print(str(decoded_labels[0]))
    metric = Bleu()
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}


    return result

def bleu_score(output_dir):
    prediction_file = os.path.join(output_dir, "generated_predictions.txt")
    with open(prediction_file, "r") as f:
        summaries = f.readlines()

    ref_file = os.path.join(output_dir, "refs.txt")
    with open(ref_file, "r") as f:
        refs = f.readlines()

    result = compute_metrics(summaries, refs)
    # print(result)
  
    output_bleu_file = os.path.join(output_dir, "BLEU_remove<s>.txt")
    with open(output_bleu_file, "w") as writer:
        json.dump(result,writer,indent=4)
    with open('result_bleu_500.txt',"a") as f3:
        f3.write(output_bleu_file+"\t")
        f3.write(json.dumps(result))
        f3.write("\n")


if __name__ == "__main__":
    # path = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data3"
    # for dir in os.listdir(path):
    #     # output_dir = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data3/checkpoint-30000/block_5"
    #     output_dir = os.path.join(path, dir,"block_3")
    #     print(output_dir)
    #     if os.path.exists(output_dir):
    #         bleu_score(output_dir)

    # output_dir = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data4_norefs/checkpoint-2600/block_4"
    # bleu_score(output_dir)
    output_dir = "/users7/kzhu/science_summ/paper2structure/test_rouge/"

    # ref_res = "/users7/kzhu/science_summ/paper2structure/test_rouge/test_label.txt"
    # dec_res = "/users7/kzhu/science_summ/paper2structure/test_rouge/test_pred.txt"
    dec_res = "/users7/kzhu/science_summ/Target-aware-RWG-main/src/results/tad.40000.candidate"
    ref_res = "/users7/kzhu/science_summ/Target-aware-RWG-main/src/results/tad.40000.gold"

    with open(ref_res,"r") as f:
        refs = f.readlines()
    refs = [ref.replace("<eos>","").strip() for ref in refs] 

    with open(dec_res,"r") as f:
        preds = f.readlines()
    preds = [pred.replace("<eos>","").strip() for pred in preds] 

    compute_metrics_title(output_dir,preds, refs)