import os
import json
import re
import random 
import argparse
import regex as re
from tqdm import tqdm

def clean_outline(title):
    title = title.replace("\n"," ").replace("\t"," ")
    pattern2 = re.compile("([\(&]*[0-9ivxIVA-Fa-f][.|\)-]*)*(-[a-kA-K]*)*([0-9.])*\s")
    match = re.match(pattern2,title.strip())  
    if match:
        # print(match.group(),tmp_line.replace(match.group(),""))
        title = title.replace(match.group(),"").strip()
    # if title.
    return title.strip()


def combine_data(survey_info_path, survey_path, ref_info_path, combine_save_path):
    if not os.path.exists(combine_save_path):
        os.makedirs(combine_save_path)

    survey_info = json.load(open(os.path.join(survey_info_path,"survey_info.json")))
    for outline_file in os.listdir(survey_path):
        if ".json" not in outline_file:
            continue
        # print(os.path.join(outline_path,outline_file))


        outline = json.load(open(os.path.join(survey_path,outline_file)))
        outline_list = []
        pattern = re.compile(r'S[\w\d.]+')

        for line in outline:
            res = re.findall(pattern, line['id'])
            if len(res)>0:
                level = len(res[0].split("."))
                if level > 3:
                    continue
            else:
                continue
            heading = clean_outline(line['title'])
            outline_list.append((level,heading))
        if len(outline_list) < 5:
            continue
        
        try:
            ref_dic_list = json.load(open(os.path.join(ref_info_path,outline_file)))["data"]
            ref_list = []

            for ref_dic in ref_dic_list:
                ref = ref_dic['citedPaper']
                ref_list.append({'paperId':ref['paperId'], 'title':ref['title'], 'abstract':ref['abstract']})

            ref_list_new = []
            for ref in ref_list:
                if ref["abstract"] is not None:
                    ref_list_new.append(ref)

            if len(ref_list_new) < 5:
                continue

            arxiv_id = outline_file[:-4].replace("ovo","/")
            title = survey_info[arxiv_id]['title']
            categories = survey_info[arxiv_id]['categories']
            survey_dic = {"title":title, "catagories":categories, "outline":outline_list, "refs":ref_list_new}
            save_path = os.path.join(combine_save_path, outline_file)
            json.dump(survey_dic,open(save_path,'w',encoding='utf-8'),indent=2)

        except Exception as e:
            print(e,outline_file)


def split(combine_path, split_path):

    dataset = []

    for outline_file in tqdm(os.listdir(combine_path)):

        file_dic = json.load(open(os.path.join(combine_path,outline_file)))
        # category = file_dic['catagories']
        title = file_dic['title']
        outline = file_dic['outline']
        ref_list_old = file_dic['refs']

        query = f"<s> <title> {title} </s>"
        source = ""
        for ref in ref_list_old:
            source += f'<s>{ref["abstract"]}</s>'
        target = ""
        for level,heading in outline:
            # target += f"<s{level}> {heading} "
            if level < 4:
                tmp_heading = clean_outline(heading)
                # print(tmp_heading)
                target += f"<s{level}> {tmp_heading} "


        dataset.append({"query":query, "target":target, "source":source})
    
    random.shuffle(dataset)
    # print(dataset)
    count = max(int(len(dataset)/10),1)
    test = dataset[:count]
    valid = dataset[count:count*2]
    train = dataset[count*2:]
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    with open(split_path+"/test.json",'w',encoding='utf-8') as f:
        for t in test:
            f.write(json.dumps(t)+"\n")

    with open(split_path+"/val.json",'w',encoding='utf-8') as f:
        for t in valid:
            f.write(json.dumps(t)+"\n")
    
    with open(split_path+"/train.json",'w',encoding='utf-8') as f:
        for t in train:
            f.write(json.dumps(t)+"\n")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_path", type=str, default="./data/survey_outline")
    parser.add_argument("--ref_info_path", type=str, default="./data/ref_info") 
    parser.add_argument("--combine_save_path", type=str, default='/users7/kzhu/Datasets/arxiv/23.04.23/combine_data/pro_combine3')
    parser.add_argument("--split_save_path", type=str, default='./data/split_path')
    parser.add_argument("--survey_info_path", type=str, default="./data/survey_info")

    args = parser.parse_args()

    # combine_data(args.survey_info_path, args.survey_path, args.ref_info_path, args.combine_save_path)
    split(args.combine_save_path, args.split_save_path)