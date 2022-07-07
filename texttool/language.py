'''
Created on 2022年2月23日

@author: Hsiao-Chien Tsai
'''
import collections
from utils import filetools
from click.types import File
def ngram(text,min,max): 
    res = []
    for n in range(min,max+1) :
        res += [text[i:i+n] for i in range(len(text)-(n-1))]
    return res
sentences = "測試123"

def sentences_ngram_dist(sentences):
    freq_count = {}
    for s in sentences :
        n_gram_word = ngram(s,min=2,max=len(s))
        for w in n_gram_word :
            if w in freq_count :
                freq_count[w] = freq_count[w] + 1 
            else :
                freq_count[w] = 1
    #sort by frequency
    
    freq_count = collections.OrderedDict(sorted(freq_count.items(), key=lambda x: -x[1]))
    return freq_count

def get_top_sentences_ngram(sentences, top_k, ascendant):
    
    freq_count = sentences_ngram_dist(sentences) 
    #技巧: 找詞頻小的stop_word sequence 反而能取更無意義的sequence
    if ascendant :
        freq_count = collections.OrderedDict(sorted(freq_count.items(), key=lambda x: x[1]))
    res = {}
    c=0 
    for k,v in freq_count.items():
        res[k] = v
        c += 1
        if c >= top_k :
            break
    return res

def file_top_ngram(file_path,top_k, ascendant):
    lines = []
    with open(file_path,"r",encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    top_ngrams = get_top_sentences_ngram(lines, top_k, ascendant)
    #for k,v in top_ngrams.items() :
    #    print("{} --> {}".format(k,v))
    return list(top_ngrams.keys())

def create_stop_labels(file_path,dummy_signal,top_k=50):
    # 為特殊標籤，代表無意義的 dummy text    
    res = [[n, dummy_signal] for n in file_top_ngram(file_path,top_k,ascendant=False)]
    
    return res

if __name__ == "__main__" :        
    stop_word_path = "../config/sys/stop_word_tw.txt" 
    file_top_ngram(stop_word_path,top_k=30, ascendant=False)   
        