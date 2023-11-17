import os
import json
import requests
import argparse
from tqdm import tqdm
import time
from threading import Thread
parser = argparse.ArgumentParser()
import urllib3
urllib3.disable_warnings()



def craw_html(survey_html_text, survey_info_path, thread_num, max_num):

    if not os.path.exists(survey_html_text):
        os.makedirs(survey_html_text)
    papers = []
    have_done = os.listdir(survey_html_text)
    survey_info = json.load(open(os.path.join(survey_info_path,"survey_info.json")))

    id_list = list(survey_info.keys())

    for id in id_list:
        if id.replace("/","ovo").strip() + '.txt' not in have_done:
            papers.append(id.strip())

    threads = []
    papers = papers[:min(len(papers),max_num)]
    batch = int(len(papers)/thread_num) + 1
    for i in range(thread_num):
        threads.append(Thread(target=run, args=(papers[i*batch: (i+1)*batch], survey_html_text)))
    for idx, t in enumerate(threads):
        print(idx, ' start')
        t.start()
    for t in threads:
        t.join()


def run(paper_list,savepath):
    for url in tqdm(paper_list):
        # url = paper[:-4]
        try:
            text = requests.get("https://ar5iv.labs.arxiv.org/html/" + url, verify=False).text
            with open(os.path.join(savepath,url.replace("/","ovo") + '.txt'), 'w', encoding='utf-8') as fp:
                fp.write(text)
                time.sleep(2)
        except:
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_info_path", type=str, default="./data/survey_info")
    parser.add_argument("--survey_html_path", type=str, default="./data/survey_html")
    parser.add_argument("--worker", type=int, default=10)
    parser.add_argument("--max_num", type=int, default=10)

    args = parser.parse_args()
    survey_html_text = args.survey_html_path
    survey_info_path = args.survey_info_path
    thread_num = args.worker
    max_num = args.max_num

    craw_html(survey_html_text, survey_info_path, thread_num, max_num)



