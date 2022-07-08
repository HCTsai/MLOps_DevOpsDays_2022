'''
Created on 2022年2月17日

@author: Hsiao-Chien Tsai
'''

import sys
from opencc import OpenCC
from os.path import dirname, abspath
project_root = dirname(dirname(abspath(__file__))) # /h
sys.path.append(project_root)

from text2vec import SBert, semantic_search
import collections
from utils import filetools
from nlp import language
#
#performance analysis
import time
import os

#text = 'Hi 我在網咖需要軟體與滑鼠資訊安全'
#print(cc.convert(text))
# 檢查是否可以使用GPU
# import torch 
# print(torch.cuda.is_available())

#初始化編碼器，第一次使用時，會自動到hugginface網站下載模型 
from model_manager import model_inference

xgb_swot = model_inference.model_xgb
#os.environ['TRANSFORMERS_CACHE'] = '/blabla/cache/'
model_name = "hfl/chinese-roberta-wwm-ext" 
#model_name = "ckiplab/bert-base-chinese"
print ("初始化語言模型:{}".format(model_name ))
embedder = SBert(model_name)
#embedder = SBert("hfl/chinese-macbert-base")
#embedder = SBert("hfl/chinese-macbert-large")
#embedder = SBert("shibing624/text2vec-base-chinese")
#embedder = SBert("hfl/chinese-roberta-wwm-ext-large")
dummy_signal = "#"
stop_word_path = "../config/swot/sys/stop_word_tw.txt"
dummy_corpus_label = language.create_stop_labels(stop_word_path, dummy_signal, top_k=30)
# load xgb SWOT model
enable_classifier_filter = True 

# 繁簡轉換
# s2twp  tw2sp
cc = OpenCC('tw2sp')
#print ("dummy corpus, size:{}".format(len(dummy_corpus)))
def label_to_desc(strategy_type):
    label_config = "../config/swot/label/{}.txt".format(strategy_type)
    label_desc = filetools.file_to_dict(label_config)
    return label_desc
def get_synonyms(text):
    aug_text = cc.convert(text)
    return aug_text 

def dict_to_table(dic, strategy_type):
    label_desc = label_to_desc(strategy_type)
    table_list = []
    type_order = list(strategy_type)
    id = 0 
    type_score = {}
    for t in type_order :
        sum_score = 0 
        for k,v in dic[t].items() :
            element = {}
            #element["id"] = id 
            element["label"] = t 
            element["type"] = label_desc[t] 
            element["text"] = k
            element["score"] = round(v, 3)
            sum_score += round(v, 3)
            table_list.append(element)
            id += 1
        avg = 0 
        if len(dic[t]) > 0:    
            avg = sum_score/len(dic[t])
        type_score[t] = round(avg,3)
    return table_list, type_score
def text_to_label(text, strategy_type, top_k, semantic_correction = False, dummy_label=True, demo_mode=False):
    st_q = time.time()
    #Step 1: 取得 使用者設定的 golden sample
    sys_defined_corpus = "../config/swot/sys/{}.txt".format(strategy_type)
    usr_defined_corpus = "../config/swot/usr/{}.txt".format(strategy_type)
    corpus_label = filetools.file_to_list(sys_defined_corpus)
    user_defined_label = filetools.file_to_list(usr_defined_corpus)
    #合併基本 S.W.O.T 與 用戶設定的 S.W.O.T
    corpus_label += user_defined_label
    # 建立無意義的 dummy 標籤，用來過濾無意義的句子
    
    if dummy_label :
        corpus_label += dummy_corpus_label
        
    #取得 Label 與 顯示名稱
    label_config = "../config/swot/label/{}.txt".format(strategy_type)
    label_desc = filetools.file_to_dict(label_config)
    
    #Step 3: 將文件轉換成多個短句子
    # multi_line = False  是否將多行當成一行來分析。(網頁與新聞時常用行 "\n" 來斷句)
    candidate_sentences = filetools.cut_sentences(text, multi_line=False)
    #過濾 不合理的句子
    candidate_sentences = [s for s in candidate_sentences 
                           if len(s)>4 and len(s)<500 and filetools.chi_ratio(s) > 0.7]
    if demo_mode :
        #選擇性取得句子最長的樣本來分析
        candidate_sentences = sorted(candidate_sentences,key=lambda k: -len(k))[:100]
    
    #print ("候選句子:{}".format(len(candidate_sentences)))
    
    #初始化分析結果 
    strategy_dict = {} #4 dictionary
    for k, v in label_desc.items() :
        strategy_dict[k] = {}
    
    if len(candidate_sentences) == 0 :
        return strategy_dict
    
    #Step 4: 對每句話，計算每句話到 Label 的距離
    # Query = P,E,S,T
    # candidates = 句子
    cls_corpus = [cl[0] for cl in corpus_label]
    cls_embeddings = embedder.encode(cls_corpus)
    aug_cls_embeddings = [] 
    if semantic_correction :
        # Label 語意 Augment，用來做 semantic correction
        aug_corpus_label = []   # 同語意的擴增標籤
        aug_candidate_sentences = [] # 同語意的擴增句子
        for c in corpus_label :
            #[corpus,label] ---> [corpus_synonym,label]
            aug_corpus_label.append([get_synonyms(c[0]),c[1]])
        #print ("data augment:{} -->{}".format(corpus_label[0],aug_corpus_label[0]))
        for c in candidate_sentences :
            aug_candidate_sentences.append(get_synonyms(c))
        aug_cls_embeddings = embedder.encode([cl[0] for cl in aug_corpus_label])
    
    # 對句子進行 encoding
   
    query_embeddings = embedder.encode(candidate_sentences)
    aug_query_embeddings = []
    if semantic_correction :
        # 對擴增句子進行 encoding
        aug_query_embeddings = embedder.encode(aug_candidate_sentences)
    
    #語意搜尋  用戶輸入句子 ----------> Label
    query_hits = semantic_search(query_embeddings, cls_embeddings, top_k=5)
    #擴增句子語意搜尋  擴增輸入句子 ----------> Label
    aug_query_hits = []
    if semantic_correction :
        aug_query_hits = semantic_search(aug_query_embeddings, aug_cls_embeddings, top_k=5)
    same_ans = 0
    correct_count = 0
    
    if enable_classifier_filter and strategy_type == "SWOT":
        xgb_predict_idx = xgb_swot.predict(query_embeddings)
        xgb_predict_prob = xgb_swot.predict_proba(query_embeddings)
        xgb_predict_label = [model_inference.idx_to_label[idx] for idx in xgb_predict_idx ]
        for i , label in enumerate(xgb_predict_label) :
            if label in strategy_dict:
                query_text = candidate_sentences[i].replace("\n","") 
                prob = round(float(xgb_predict_prob[i][xgb_predict_idx[i]]),3)
                if prob > 0.5 and len(query_text) < 30:
                    strategy_dict[label][query_text] = prob # 紀錄 S.W.O.T 分組底下的 query 與 分數
    else :    
        for i, hits in enumerate(query_hits) :
            query_text = candidate_sentences[i].replace("\n","") 
            # hits  [{'corpus_id': 17, 'score': 0.71844482421875}]
            # 用戶輸入句子
            hit = hits[0]  # 只找top1 句離最接近的，分數最高的
            corpus_id = hit["corpus_id"]  # 第 n 筆 corpus
            corpus_score = hit["score"]
            hit_label = corpus_label[corpus_id][1] # 取得label
            
            # 語意糾錯
            if semantic_correction :
                aug_hit_label = ""
                # 擴增句子
                aug_hit = aug_query_hits[i][0] # top1 score
                aug_corpus_id = aug_hit["corpus_id"]
                aug_corpus_score = aug_hit["score"]
                aug_hit_label = aug_corpus_label[aug_corpus_id ][1] # 取得label
                
                
                if hit_label == aug_hit_label :
                    same_ans += 1
                
                else : # 一句話 的 同語意 句子，卻對應到了不同 Label，進行語意Correction
                    # 如果候選句子，有更接近Label語意
                    #print ("{}:\t{} {}------->{} {}".format(query_text, hit_label, corpus_score , aug_hit_label, aug_corpus_score))
                    if semantic_correction and aug_corpus_score > 0.65 and (aug_corpus_score - corpus_score) > 0.001  :
                            #semantic correction
                            #print ("{}:\t{}--correct-->{}".format(query_text, hit_label, aug_hit_label))
                            #語意 Correction 修正 
                            hit_label = aug_hit_label
                            corpus_score = aug_hit["score"]
                            correct_count += 1
                        #print ("{}:{}   {}:{}".format(hit_label,corpus_score,aug_hit_label,aug_corpus_score))
            
            
            if dummy_label :
                if hit_label == dummy_signal :
                    #print ("無意義句子過濾 {}: {}".format(hit_label, query_text))
                    continue
            if hit_label in strategy_dict :
                strategy_dict[hit_label][query_text] = corpus_score # 紀錄 S.W.O.T 分組底下的 query 與 分數
    
    
    # Step 5: 輸出分析結果
    '''
    print ("classifier consistency ratio:{}".format(same_ans/len(query_hits)))
    print ("classifier correction ratio:{}".format(correct_count/len(query_hits)))
    '''
    
    for s, d in strategy_dict.items() :
        #dictionary to score list with sorting
        d = sorted(d.items(), key=lambda x: -x[1])
        strategy_dict[s]=collections.OrderedDict(d[:top_k])
        '''
        print (label_desc[s]+":"+"\n---------------")
        for k, v in d[:top_k]:
            print ("{} (score:{:.4f})".format(k,v))
        '''
    et_q = time.time()
    #print ("text_to_label time elapse:{} Sec.".format(et_q-st_q))    
    
    return strategy_dict

if __name__ == "__main__" :
    text = "放眼世界市場"
    strategy_dict = text_to_label(text, strategy_type="PEST", top_k=5, semantic_correction = True, dummy_label=True)
    
    
