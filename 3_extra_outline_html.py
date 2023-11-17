from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
import argparse

def extra_outline(raw_html_path, pro_outline_path):

    papers = os.listdir(raw_html_path)

    if not os.path.exists(pro_outline_path):
        os.makedirs(pro_outline_path)

    for name in os.listdir(pro_outline_path):
        if name in papers:
            papers.remove(name)

    len_start = len(papers)

    for paper in tqdm(papers):
        try:
            with open(os.path.join(raw_html_path, paper)) as fp:
                read_lines = fp.readlines()
                text = "".join(read_lines)
                soup = BeautifulSoup(text, 'html.parser')
                sections = soup.find_all('section')
                section_info = []
                for section in sections:
                    sec_id = section['id']
                    sec_texts = section.text.split('\n')
                    sec_text = ""
                    for sec_txt in sec_texts:
                        if len(sec_txt.strip()) > 0:
                            sec_text = sec_txt
                            break
                    ret = {'id': sec_id, 'title': sec_text}
                    section_info.append(ret)
                if len(section_info) > 0:
                    json.dump(fp=open(os.path.join(pro_outline_path, paper), 'w'), obj=section_info, indent=2)     
        except:
            continue

    for name in os.listdir(pro_outline_path):
        if name in papers:
            papers.remove(name)

    with open(os.path.join(pro_outline_path, "error.log"),'w',encoding='utf-8') as f:
        f.write("\n".join(papers))
        
    print(f"Processed {len_start - len(papers)} papers outline")
    print(len(papers), "papers' outline failed")

    # , len(os.listdir(pro_outline_path))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_html_path", type=str, default="./data/survey_html")
    parser.add_argument("--survey_outline_path", type=str, default="./data/survey_outline")
    args = parser.parse_args()
    
    raw_html_path = args.survey_html_path
    pro_outline_path = args.survey_outline_path
    extra_outline(raw_html_path, pro_outline_path)


