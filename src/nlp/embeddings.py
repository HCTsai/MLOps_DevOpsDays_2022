'''
Created on 2022年6月15日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from text2vec import SBert
model_name = "hfl/chinese-roberta-wwm-ext" 
embedder = SBert(model_name)

def sentences_to_vectors(sentences):
    return embedder.encode(sentences)
    