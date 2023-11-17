import os
import requests
import time
from tqdm import tqdm
import json
from threading import Thread
import argparse

# "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{}/references?fields=paperId,title,abstract,url,contexts,intents,isInfluential&offset=0&limit=500


def craw_ref(save_path, survey_info_path, thread_num):
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    have_craw = os.listdir(save_path)

    dois = []
    survey_lis = os.listdir(survey_info_path)
    for survey in survey_lis:
        if ".json" in survey and survey not in have_craw:
            dois.append(survey[:-5].replace("ovo","/"))

    threads = []
    batch = int(len(dois)/thread_num) + 1
    for i in range(thread_num):
        threads.append(Thread(target=run, args=(save_path,dois[i*batch: (i+1)*batch])))
    for idx, t in enumerate(threads):
        print(idx, ' start')
        t.start()
    for t in threads:
        t.join()



def run(save_path, doi_list):

    for doi in tqdm(doi_list):
        url = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{}/references?fields=title,abstract&offset=0&limit=500".format(doi)
        try:
            response = requests.get(url, headers=headers)
            with open(os.path.join(save_path, doi.strip().replace("/", "ovo") + '.json'), "wb") as code:
                code.write(response.content)
        except Exception as e:
            print(e)
            with open(os.path.join(save_path,"error.log"), 'a') as err:
                err.write(save_path + doi.strip().replace("/", "ovo") + '.json'+"\t")
                err.write(doi + ':' + doi.strip() + '\n')
        time.sleep(1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--survey_outline_path", type=str, default="./data/survey_outline")
    parser.add_argument("--ref_info_path", type=str, default="./data/ref_info") 
    parser.add_argument("--worker", type=int, default=10)

    args = parser.parse_args()

    headers = {"x-api-key": args.api_key}
    survey_info_path = args.survey_outline_path
    save_path = args.ref_info_path
    thread_num = args.worker
    craw_ref(save_path, survey_info_path, thread_num)

    # print(path, ' end.')