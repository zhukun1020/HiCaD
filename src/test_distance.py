"""

版权声明：本文为CSDN博主「kuokay」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_45066628/article/details/123170726

#关于 Levenshtein 所有函数的用法和注释
apply_edit()  #根据第一个参数editops（）给出的操作权重，对第一个字符串基于第二个字符串进行相对于权重的操作
distance() #计算2个字符串之间需要操作的绝对距离
editops() #找到将一个字符串转换成另外一个字符串的所有编辑操作序列
hamming() #计算2个字符串不同字符的个数，这2个字符串长度必须相同
inverse() #用于反转所有的编辑操作序列
jaro() #计算2个字符串的相识度，这个给与相同的字符更高的权重指数
 
jaro_winkler() #计算2个字符串的相识度，相对于jaro 他给相识的字符串添加了更高的权重指数，所以得出的结果会相对jaro更大（%百分比比更大）
matching_blocks() #找到他们不同的块和相同的块，从第六个开始相同，那么返回截止5-5不相同的1，第8个后面也开始相同所以返回8-8-1，相同后面进行对比不同，最后2个对比相同返回0
median() #找到一个列表中所有字符串中相同的元素，并且将这些元素整合，找到最接近这些元素的值，可以不是字符串中的值。
median_improve() #通过扰动来改进近似的广义中值字符串。
 
opcodes() #给出所有第一个字符串转换成第二个字符串需要权重的操作和操作详情会给出一个列表，列表的值为元祖，每个元祖中有5个值
    #[('delete', 0, 1, 0, 0), ('equal', 1, 3, 0, 2), ('insert', 3, 3, 2, 3), ('replace', 3, 4, 3, 4)]
    #第一个值是需要修改的权重，例如第一个元祖是要删除的操作,2和3是第一个字符串需要改变的切片起始位和结束位，例如第一个元祖是删除第一字符串的0-1这个下标的元素
    #4和5是第二个字符串需要改变的切片起始位和结束位，例如第一个元祖是删除第一字符串的0-0这个下标的元素，所以第二个不需要删除
 
quickmedian() #最快的速度找到最相近元素出现最多从新匹配出的一个新的字符串
ratio() #计算2个字符串的相似度，它是基于最小编辑距离
seqratio() #计算两个字符串序列的相似率。
setmedian() #找到一个字符串集的中位数(作为序列传递)。 取最接近的一个字符串进行传递，这个字符串必须是最接近所有字符串，并且返回的字符串始终是序列中的字符串之一。
setratio() #计算两个字符串集的相似率(作为序列传递)。
subtract_edit() #从序列中减去一个编辑子序列。看例子这个比较主要的还是可以将第一个源字符串进行改变，并且是基于第二个字符串的改变，最终目的是改变成和第二个字符串更相似甚至一样
"""

from math import dist
import Levenshtein as ls
import os
import re
import json

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    pattern = re.compile("<s[1-3]>")

    final_preds = []
    for pred in preds:
        tmp_pred = re.findall(pattern,pred)
        tmp_pred = "".join(tmp_pred)
        tmp_pred = tmp_pred.replace("<s","").replace(">","").replace(" ","")
        final_preds.append(tmp_pred)

    final_labels = []
    for label in labels:
        tmp_label = re.findall(pattern,label)
        tmp_label = "".join(tmp_label)
        tmp_label = tmp_label.replace("<s","").replace(">","").replace(" ","")
        final_labels.append(tmp_label)

    return final_preds, final_labels

def compute_TEDS(output_dir, summaries=None, refs=None):
    
    if summaries is None:
        prediction_file = os.path.join(output_dir, "generated_predictions.txt")
        with open(prediction_file, "r") as f:
            summaries = f.readlines()
    if refs is None:
        ref_file = os.path.join(output_dir, "refs.txt")
        with open(ref_file, "r") as f:
            refs = f.readlines()

    decoded_preds, decoded_labels = postprocess_text(summaries, refs)
    # TEDS(decoded_preds,decoded_labels)

    score = TEDS(decoded_preds,decoded_labels)

    output_bleu_file = os.path.join(output_dir, "TEDS_dis.txt")
    with open(output_bleu_file, "w") as writer:
        json.dump(score,writer,indent=4)
    with open('result_teds.txt',"a") as f3:
        f3.write(output_dir+"\t")
        f3.write(json.dumps(score))
        f3.write("\n")


def TEDS(str1_list,str2_list):
    assert len(str1_list) == len(str2_list)

    score_dic = {}
    # maxL=len(str1)
    # if len(str2)>maxL:
    #     maxL=len(str2)
    total_dis = 0
    total_res = 0
    total_str1_dis = 0
    total_str2_dis = 0
    for i in range(len(str1_list)):
        str1 = str1_list[i]
        str2 = str2_list[i]
        maxL = max(len(str1),len(str2))
        distan=ls.distance(str1,str2)
        teds_result=100-distan/maxL*100
        # print(str1,str2,distan,maxL,teds_result)
        # if maxL > 10:
        #     print(i)

        total_str1_dis += len(str1)
        total_str2_dis += len(str2)
        total_dis += distan
        total_res += teds_result
    # print(distan)
    
    score_dic['text_edit_distance'] = total_dis/len(str1_list)
    score_dic['text_edit_distance_score'] = total_res/len(str1_list)
    score_dic['preds_avg_len'] = total_str1_dis/len(str1_list)
    score_dic['refs_avg_len'] = total_str2_dis/len(str2_list)

    return score_dic
    # teds_result=100-distan/maxL*100
    # return teds_result


# s1 = "12211123"
# s2 = "1211123"
# TEDS(s1,s2)
if __name__ == "__main__":

    compute_TEDS("/users7/kzhu/science_summ/paper2structure/res/test_config/checkpoint-30000/block_4/min_100")

    # path = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data4_norefs"
    # for dir in os.listdir(path):
    #     # output_dir = "/users7/kzhu/science_summ/paper2structure/res/longt5_BigSurvey_16384_data3/checkpoint-30000/block_5"
    #     output_dir = os.path.join(path, dir,"block_5")
    #     print(output_dir)
    #     if os.path.exists(output_dir):
    #         try:
    #             compute_TEDS(output_dir)
    #         except:
    #             print(output_dir)