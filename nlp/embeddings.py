'''
Created on 2022年6月15日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from text2vec import SBert
from config import global_config
model_name = global_config.embedding_model_name
embedder = SBert(model_name)

def sentences_to_vectors(sentences):
    return embedder.encode(sentences)
    