"""
Created on 2021年5月3日

@author: hsiaochien.tsai
"""

import pandas as pd
import re
import string

def list_to_file(data_list,cols=[], delimiter = ",", out_file = "outfile.txt"):
    with open(out_file,"w",encoding="utf-8") as f:        
        for row in data_list : 
            line = ""
            if len(cols) == 0 :
                line = delimiter.join(row)
            else :
                c_list = []
                for c in cols :
                    c_list.append(row[c])
                line = delimiter.join(c_list)
                if c_list[-1] == "" :
                    print (row)
                    print (line)
            
            f.write(line+"\n")
def remain_chi_eng(text):
    #只保留 中文與英文
    text = re.sub(r'[^\u4e00-\u9fa5\w\?!？！。…，,]',"",text)
    return text
def get_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5\?!？！。…，,]')
    chinese = re.sub(pattern, "", text)
    return chinese
def replace_newline(text,rep=""):
    text = re.sub(r"[\r\n\t\s]+", rep, text)
    text = re.sub(r"\\n", rep, text)
    return text
def clr_sentence(text):
    text = re.sub(r"[\r\n\t\s]+", "", text)
    #TO-DO:去除stop-word
    stop_words = ["並","同時","也","以及","不僅","否則","而","在"]
    for word in stop_words :
        text = re.sub(word,"",text) 
    
    chi_puncutaions ="！？。，｡＂＃＄％＆＇＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.✶"
    head_punctuations = "".join(string.punctuation) + chi_puncutaions
    #移除 所有開頭標點符號 
    text = re.sub(r'^[{}]+'.format(head_punctuations),"",text) 
    #結尾標點符號統一表示 
    tail_puncutaions ="，！？。，｡＂＃＄％＆＇＊＋，－／：；＝＠＼＾＿｀｜～､、〃〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.✶"
    reg_tail = r"[{}]+$".format(tail_puncutaions)
    text = re.sub(reg_tail,"",text) 
    #移除符號
    chi_para ="＜＞［］｛｜｝｟｠｢｣〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰‘'‛“”„‟…"
    paras = "".join(chi_para)
    reg_para = r"[{}]".format(paras)
    text = re.sub(reg_para,"",text) 
    
    return text

def cut_sentences(content, multi_line=False, sent_len=7):
    # 結束符號，包含中文和英文的
    if (multi_line) : #多行一起看
        content = replace_newline(content," ") #須注意: 換行不一定是句子結尾
    #分隔符號，須注意 '.' 可能是數學小數點，可能是英文句點 
    #分隔符號，須注意 ','  2,664億元
    #,"、"
    seg_flag = ['?', '!', '？', '！', '。', '…',"，","\n","；",",","「","」"]
    content_len = len(content)
    
    sentences = []
    temp_sentence = ""
    # 拼接句子
    for idx, char in enumerate(content):
        
        temp_sentence += char

        # 判斷是否已經到了最後一位
        if (idx + 1) == content_len:
            sentences.append(clr_sentence(temp_sentence))
            break
            
        # 判斷此字符是否爲分隔符號
        if char in seg_flag  :
            # 再判斷下一個字符是否爲結束符號，如果不是結束符號，則切分句子
            # 如果句子太短，繼續拼接。
            next_idx = idx + 1
            if not content[next_idx] in seg_flag and len(temp_sentence) >= sent_len:
                sentences.append(clr_sentence(temp_sentence))
                temp_sentence = ""
                
    return sentences
def file_to_sentence_list(file_path, multi_line=False):
    res = []
    with open(file_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        lines = remove_non_chinese(lines)
        sentences = cut_sentences("".join(lines), multi_line)
        for s in sentences : 
            if len(s)>3 : # 結尾可能是少於 sent_len的句子
                res.append(s)
    return res
            
def file_to_list(file_path, delimiter = ",", ignore="#"):
    res = []
    with open(file_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines :
            if not line.startswith(ignore) :
                data_list = line.strip().split(sep=delimiter)
                res.append(data_list) 
                
    return res

def file_to_dict(file_path, delimiter = ",", ignore="#"):
    res = {}
    with open(file_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines :
            if not line.startswith(ignore) :
                data_list = line.strip().split(sep=delimiter)
                if len(data_list) == 2 :
                    res[data_list[0]] = data_list[1]
                else :
                    print ("file: {} format error".format(file_path))
    return res

def file_to_text(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        #過濾
        lines = remove_non_chinese(lines)
        return " ".join(lines)    
    
def pd_dataframe_to_list(df):
    #samples=df[:]
    df.fillna('', inplace=True)
    sample_list = df.values.tolist()
    return sample_list 
def excel_to_list(file_path, cols=None, type=None, sheet_name = None):
    if type == None :
        ext = file_path.split(sep=".")[-1]
        if ext == "xls" or ext == "xlsx" :
            type = ext
    engine = None    
    if (type == "xlsx") :
        engine = "openpyxl"  
    df = None
    if sheet_name == None :
        df= pd.read_excel(file_path,usecols=cols, engine=engine)
    else :
        df = pd.read_excel(file_path,usecols=cols, engine=engine, sheet_name=sheet_name)
    
    sample_list = pd_dataframe_to_list(df)
    return sample_list 

def get_excel_column(file_path, col, type = None):    
    sample_list = excel_to_list(file_path, cols=[col], type=None)
    flat_list = [s[0] for s in sample_list]  
    return flat_list 
def is_chinese(text):
    if re.findall(r'[\u4e00-\u9fff]+', text):
        return True 
    return False
def remove_non_chinese(in_lines):
    '''
    移除沒有包含任何中文的句子
    '''
    out_lines = []
    for text in in_lines :
        if is_chinese(text) :
            out_lines.append(text)
            
    return out_lines        
def chi_ratio(text):
    chi_count = 0
    for t in text :
        if is_chinese(t) :
            chi_count += 1 
    return chi_count/len(text)
if __name__ == '__main__' :
    print(chi_ratio("展望2022年"))
    pass
    