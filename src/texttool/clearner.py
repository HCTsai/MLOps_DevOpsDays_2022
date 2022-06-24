'''
Created on 2021年3月30日

@author: hsiaochien.tsai
'''
import re
import string


def remove_newline(text):
    text = re.sub(r"[\r\n\t\s]+", " ", text)
    text = re.sub(r"\\n", "", text)
    #text = re.sub(r"[\n\t\s]*", "", text)
    return text
def remove_space(text):
    text = re.sub(r"[\r\n\t\s]+", "", text)
    #text = re.sub(r"[\n\t\s]*", "", text)
    return text
def remove_punctuation(text):    
    #移除 開頭標點符號 
    text = re.sub(r'^[\s\-\*_%]+','',text)  
    #移除 結尾標點符號 
    text = re.sub(r'[\s\-\*_%]+$','',text)   
    #移除 中間標點符號。 保留 空白 -_ / *  以外的所有標點符號
    text = re.sub(r'[^\w\s\-\*_%]','',text)
    
    return text

def strip_punctuation(text): 
    """
    移除開頭與結尾符號
    """
    
    chi_puncutaions ="，！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
    mid_punc ="\(\)＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
    punctuations = "".join(string.punctuation) + chi_puncutaions
    #移除 所有開頭標點符號 
    text = re.sub(r'^[{}]+'.format(punctuations),'',text)  
    #移除 所有結尾標點符號 
    text = re.sub(r'[{}1234567890]+$'.format(punctuations),'',text) 
    #移除中間 中文標點符號 ，原則上應該移除逗點，句點以外的所有標點  
    text = re.sub(r'[{}]+'.format(mid_punc),' ',text) 
    #text = re.sub(r'[{}]+'.format(punctuations),',',text) 
    #分隔符號改逗點
    text = text.replace("|","。")
      
    return text

def remain_chi_eng(text):
    #只保留 中文與英文
    text = re.sub(r'[^\u4e00-\u9fa5\w]','',text)
    return text


def remove_ids(text):
    ids = re.findall(r'[a-zA-z\d]{6,7}[\-－][a-zA-Z\d]{2,3}', text, re.IGNORECASE)
    for word in ids:
        text = text.replace(word, "")    
    return text

def remove_date_ratio(text):
    reg_pattern = r"\-?[\d]+[/\\][\d]+\s*"
    text = re.sub(reg_pattern,"",text) 
    return text

def remove_stop_words(text):
    # *4
    reg_pattern = r"([\*#]\s*[\d]+)"
    founds = re.findall(reg_pattern, text, re.IGNORECASE)
    # X月份    
    reg_pattern = r"[一二三四五六七八九十0-9]+\s*月份?"
    founds += re.findall(reg_pattern, text, re.IGNORECASE)
    
    reg_pattern = r"[0-9]+[\*X#][0-9]+\s*月份?"
    founds += re.findall(reg_pattern, text, re.IGNORECASE)
    
    #print (founds)
    for word in founds :
        text = text.replace(word, "")             
    return text
def remove_units(text, units):
    for unit in units :
        reg_pattern = r"(\-?[\d]*/?[\d]+\s*{})".format(unit)
        text = re.sub(reg_pattern,'',text)
        reg_pattern = r"([一二三四五六七八九十0-9兩百]+\s*{})".format(unit)
        text = re.sub(reg_pattern,'',text)
    return text
def get_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', text)
    return chinese

def clear_html_tags(text):
    text = re.sub('<[^>]*>', '', text)
    return text
def clear_splitter(text):
    #分隔符號改逗點
    text = text.replace("|","。") 
    return text

'''
text ="No Backlight"
print (clear_caerb_title(text))
'''