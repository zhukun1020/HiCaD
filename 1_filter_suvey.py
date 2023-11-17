import json
import re
import os
import argparse


def filter_title(tmp_title):

    pattern = ["a survey","a review","research review", "review and perspectives",": review",": survey","- survey","- review",
            "review of", "review on", " review:", "-review:", "an overview"]
        # ,": a",]

    # pattern = ["survey","review","overview"]
               
    pattern2 = ["and", "survey","systematic","comprehensive","brief","literature","short","concise","research","introductory","topical",\
                "critical","comparative","tutorial","theory","selective","experimental","historical","state-of-the-art","state of the art","holistic",\
                "(short)","in-depth","synthetic","categorical","focused","forward-looking","creative","bibliographic",\
                "instructive","compact","statistical","rapid","quantitative","selected","rapid","perspective","unified",\
                "biased","constructive","modern","status","contextual","conceptual","empirical","mini","technical","contemporary"]
    # "tertiary","scoping"
    for p in pattern2:
        pattern.append(p+" survey")
        pattern.append(p + " review")
        pattern.append(p + " overview")

    
    title = tmp_title.lower().replace("\n","")
    if "book review" in title or "comments" in title:
        return False
    
    for p in pattern:        
        if p in title:
            return True
        tmp = title.split(": a")
        if len(tmp) > 1 and ("survey" in tmp[-1] or "review" in tmp[-1]):
            return True

    if title.startswith("survey") or title.startswith("review"):
        return True

    return False


def select_withtitle(filepath, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    survey_list = []
    count = 0
    paper_info_dic = {}
    with open(filepath,'r',encoding='utf-8') as f:
        line = f.readline()
        while line != "":
            count+=1
            if count%10000 == 0:
                print(f'{count} have processed!')
            paper_dic = json.loads(line)

            if filter_title(paper_dic['title']) and "astro-ph" not in paper_dic["categories"]:
                survey_list.append(line.strip()) 
                paper_info_dic[paper_dic['id']] = {"id":paper_dic["id"],"categories":paper_dic["categories"],"title":paper_dic["title"].replace("\n","")}

            line = f.readline()

    print(f'The total count of arxiv papers: {count}')
    print(f'The final count of arxiv survey papers: {len(survey_list)}')

    with open(os.path.join(savepath,"survey.json"),"w",encoding='utf-8') as f:
        f.write("\n".join(survey_list))
    
    with open(os.path.join(savepath,"survey_info.json"),"w",encoding='utf-8') as f:
        json.dump(paper_info_dic,f,indent=2)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxivmeta_path", type=str, default="./data/arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--survey_info_path", type=str, default="./data/survey_info")
    args = parser.parse_args()

    select_withtitle(args.arxivmeta_path, args.survey_info_path)
