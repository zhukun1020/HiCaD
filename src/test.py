from test_bleu import compute_metrics_label, compute_metrics_title
from test_distance import compute_TEDS
import os
from test_rouge import total_control_onlytitle, total_control_onlytitle_huggingface
import time
import re
import json

def remove_titlelabel(outline_lis):
    pattern = re.compile("<s[1-3]>")
    pattern2 = re.compile("([\(&]*[0-9ivxIVA-Fa-f][.|\)-]*)*(-[a-kA-K]*)*([0-9])*\s") 
    final_outline_lis = []

    for outline in outline_lis:

        target = outline
        
        labels = re.findall(pattern, target)
        target_lis = target.strip().replace("<s1>","\t").replace("<s2>","\t").replace("<s3>","\t")
        target_lis = target_lis.split("\t")[1:]
        # print(target,labels,target_lis)
        assert len(labels) == len(target_lis)

        final_target = ''

        for index in range(len(target_lis)):

            tmp_line = target_lis[index].strip()
            if tmp_line == "":
                continue
            match = re.match(pattern2,tmp_line)
            # print(tmp_line)
            if match:
                # print(match.group(),tmp_line.replace(match.group(),""))
                tmp_line = tmp_line.replace(match.group(),"").strip()
            
            final_target += fr"{labels[index]} {tmp_line} "
        final_outline_lis.append(final_target)

    # print(final_outline_lis)
    return final_outline_lis
        
def remove_titlelabelpattern(outline_lis):
    pattern = re.compile("<s[1-3]>")
    pattern2 = re.compile("([\(&]*[0-9ivxIVA-Fa-f][.|\)-]*)*(-[a-kA-K]*)*([0-9])*\s") 

    pattern3_lis = ["introduction", "methodology", "results", "discussion", "conclusion", "references",\
                    "preliminaries", "conclusions", "background", "methods", "acknowledgments", "related work"]
    final_outline_lis = []

    for outline in outline_lis:

        target = outline
        
        labels = re.findall(pattern, target)
        target_lis = target.strip().replace("<s1>","\t").replace("<s2>","\t").replace("<s3>","\t")
        target_lis = target_lis.split("\t")[1:]
        # print(target,labels,target_lis)
        assert len(labels) == len(target_lis)

        final_target = ''

        for index in range(len(target_lis)):

            tmp_line = target_lis[index].strip().lower()
        
            # print(tmp_line)  
            match = re.match(pattern2,tmp_line)  
            if match:
                # print(match.group(),tmp_line.replace(match.group(),""))
                tmp_line = tmp_line.replace(match.group(),"").strip()

            if tmp_line == "" or tmp_line in pattern3_lis:
                continue
            final_target += fr"{labels[index]} {tmp_line} "       
            # final_target += fr"{labels[index]} {target_lis[index]} "
        final_outline_lis.append(final_target)

    # print(final_outline_lis)
    return final_outline_lis

def pro_text(text_lis):
    new_lis = []
    for t in text_lis:
        t = t.replace("s1>","<s1>")
        t = t.replace("S1>","<s2>")
        for i in range(3,9):
            t = t.replace(fr"{i}>","<s3>")
        new_lis.append(t)
    return new_lis


def remove_space(summaries,refs):
    final_sum = []
    final_ref = []
    for i in range(len(summaries)):
        tmp_sum = summaries[i]
        # print("-"+tmp_sum+"-")
        if tmp_sum.strip() != "":
            final_sum.append(tmp_sum)
            final_ref.append(refs[i])
    return final_sum,final_ref


def cal():
    path = "/users7/kzhu/science_summ/paper2structure/res/led_BigSurvey_16384_data9_3nopattern_2/checkpoint-27000"
    file_dirname = ""
    output_dirname = "remove_space"

    for dir in os.listdir(path):
        dir_path = os.path.join(path,dir)
        input_dir = os.path.join(dir_path,file_dirname)     
        output_dir = os.path.join(dir_path,output_dirname)

        prediction_file = os.path.join(input_dir, "generated_predictions.txt") 
        # print(prediction_file)
        if os.path.exists(prediction_file):  
            with open(prediction_file, "r") as f:
                summaries = f.readlines()
            # summaries = remove_titlelabel(summaries)
            # summaries = pro_text(summaries)

            # summaries = remove_titlelabelpattern(summaries)

            ref_file = os.path.join(input_dir, "refs.txt")
            with open(ref_file, "r") as f:
                refs = f.readlines()
            # refs = remove_titlelabel(refs)
            # refs = remove_titlelabelpattern(refs)
            
            summaries,refs = remove_space(summaries,refs)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ref_dir = os.path.join(output_dir, "lcs_ref/")
            dec_dir = os.path.join(output_dir, "lcs_dec/")      
            total_control_onlytitle(refs, summaries, ref_dir, dec_dir)
            total_control_onlytitle_huggingface(output_dir, refs, summaries)

            # result = compute_metrics(summaries, refs)
            # print(result)
            # output_rouge_file = os.path.join(args.output_dir, "ROUGE.txt")
            # with open(output_rouge_file, "w") as writer:
            #     json.dump(result,writer,indent=4)

            compute_metrics_label(output_dir, summaries, refs)
            compute_metrics_title(output_dir, summaries, refs)
            compute_TEDS(output_dir, summaries=summaries, refs=refs)


def get_level_outline(text,level="1"):
    text_list = text.split("<s")
    final_text = ""
    for t in text_list:
        if t.startswith(level):
            final_text += " <s" + t
    return final_text




def test(prediction_filepath,ref_filepath,output_dir):
    prediction_file = os.path.join(prediction_filepath, "generated_predictions0.txt")
    with open(prediction_file, "r") as f:
        summaries = f.readlines()

    summaries = [r.replace(r.split("<s")[0],"").strip() for r in summaries]

    # ref_file = os.path.join(ref_filepath, "test.json")
    # with open(ref_filepath, "r") as f:
    #     data = f.readlines()
    # refs = [json.loads(d)["target"] for d in data]

    ref_file = os.path.join(prediction_filepath, "refs.txt")
    with open(ref_file, "r") as f:
        refs = f.readlines()
    refs = [r.strip() for r in refs]
    # summaries = [get_level_outline(s,"3").replace("<eos>","").strip() for s in summaries]
    # refs = [get_level_outline(r,"3").replace("<eos>","").strip() for r in refs]
    # refs = [r.replace("<eos>","").strip() for r in refs]
    

    output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(summaries))

    output_ref_file = os.path.join(output_dir, "refs.txt")
    with open(output_ref_file, "w") as writer:
        writer.write("\n".join(refs))   

    ref_dir = os.path.join(output_dir, "lcs_ref/")
    dec_dir = os.path.join(output_dir, "lcs_dec/")

    total_control_onlytitle(refs, summaries, ref_dir, dec_dir)
    total_control_onlytitle_huggingface(output_dir, refs, summaries)

    # result = compute_metrics(summaries, refs)
    # print(result)
    # output_rouge_file = os.path.join(args.output_dir, "ROUGE.txt")
    # with open(output_rouge_file, "w") as writer:
    #     json.dump(result,writer,indent=4)

    compute_metrics_label(output_dir, summaries, refs)
    compute_metrics_title(output_dir, summaries, refs)
    # compute_TEDS(output_dir, summaries=summaries, refs=refs)

def test_all():
    ref_path = "/users7/kzhu/science_summ/query-focused-sum/multiencoder/res_predict/multiencoder2_ondata9mx400/"
    pred_path = "/users7/kzhu/science_summ/paper2structure/res_predict/led_large_BigSurvey_16384_data9_3nopattern_3"
    out_path = "/users7/kzhu/science_summ/paper2structure/res_predict/led_large_BigSurvey_16384_data9_3nopattern_3/s3"
    for ckpt in os.listdir(pred_path):
        ckpt_dir = os.path.join(pred_path,ckpt)
        for bk in os.listdir(ckpt_dir):
            if bk.startswith("b"):
                bk_dir = os.path.join(ckpt_dir,bk)
                output_dir=os.path.join(out_path,ckpt,bk)
                print(output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                test(bk_dir,ref_path,output_dir)


def check_s2():
    pred_path = "/users7/kzhu/science_summ/paper2structure/res_predict/data_clean_moref2_title+abs_s1s2_ons1s2/checkpoint-1628-/block3/generated_predictions.txt"
    ref_path = "/users7/kzhu/science_summ/paper2structure/res_predict/data_clean_moref2_title+abs_s1s2_ons1s2/checkpoint-1628-/block3/refs.txt"
    testd_data = "/users7/kzhu/Datasets/arxiv_survey/data_clean_moref2/title+abs/givens1_gens2/test.json"
    preds = open(pred_path,"r").readlines()
    refs = open(ref_path,"r").readlines()
    
    assert len(preds)==len(refs)
    for i in range(len(preds)):
        pred = preds[i]
        ref = refs[i]
        

if __name__ == "__main__":
    # test_all()
    bk_dir="/users7/kzhu/science_summ/paper2structure/res_predict/data_clean_moref2_title+abs_decodergiventitle_gens1s2_2/checkpoint-1628/block3"
    ref_path="/users7/kzhu/science_summ/query-focused-sum/multiencoder/res_predict/multiencoder2_ondata9mx400/checkpoint-19500-/block3"
    output_dir="/users7/kzhu/science_summ/paper2structure/res_predict/data_clean_moref2_title+abs_decodergiventitle_gens1s2_2/checkpoint-1628/block3-0"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    test(bk_dir,ref_path,output_dir)
    


    # for i in range(10):
    #     print(time.time())
    # cal()

    # output_dir = "/users7/kzhu/science_summ/paper2structure/res/test_config/checkpoint-30000/block_4/min_150"
    # bleu_score(output_dir)
    # compute_TEDS(output_dir)
    # ref_res = os.path.join(output_dir,"refs.txt")
    # dec_res = os.path.join(output_dir,"generated_predictions.txt")
    # total_control_onlytitle_huggingface(output_dir, ref_res, dec_res)

