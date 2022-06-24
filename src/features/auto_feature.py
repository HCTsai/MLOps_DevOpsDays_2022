'''
Created on 2022年5月4日

@author: Hsiao-Chien Tsai
'''
# 讀取 Labeling data 產生 feature and label
# label:{0,1,2,3}  #features 768 dimensions
#
from utils import filetools
from nlp import embeddings
import numpy 
#
from sklearn.model_selection import train_test_split
from collections import Counter
from texttool import chinese_transform
from texttool import clearner
from nlp import language
import collections

# input
input_file_path = "../data/swot_4000.xlsx"
input_file_path_ext ="../data/extend_corpus.csv"
# output
output_data_file = "all_label_features"
tran_data_file = "train_label_features"
test_data_file = "test_label_features"
def write_label_feature(X,y,file_name):
    with open(file_name,"w",encoding="utf-8") as f :
        for i, features in enumerate(X):
            label_features =  [str(y[i])] + [str(f) for f in features.tolist()]
           
            out_str = ",".join( label_features )
            f.write("{}\n".format(out_str))

           
def create_label_features(label_type, label_balance):
    
    data_list = filetools.excel_to_list(input_file_path)
    ext_data_list = filetools.file_to_list(input_file_path_ext)
    #label, input
    label_idx = 0
    feature_idx = [1]
    samples = data_list[0:4000]
    
    #data clearn
    trans = chinese_transform.ChineseTransform()
    print("add extend label size:{}".format(len(ext_data_list)))
    train_samples = []
    for s in ext_data_list :
        input_text = trans.simp2trad(clearner.get_chinese(s[1]))
        if len(input_text) > 1 and len(s[0])>0 :
            train_samples.append([s[0],input_text])
    print("add corpus label")
    for s in samples :
        input_text = trans.simp2trad(clearner.get_chinese(s[1]))
        if len(input_text) > 1 and len(s[0])>0 :
            train_samples.append([s[0],input_text])
    
    label_dict =  Counter([s[0] for s in train_samples])
    print (label_dict)
    # label balance
    # 以最少的兩個 Label做為增加依據
    if label_balance:
        label_dict =  {k: v for k, v in sorted(label_dict.items(), key=lambda item: -item[1])}
        augment_count = (label_dict[list(label_dict.keys())[-2]] - label_dict[list(label_dict.keys())[-1]])/2  
        dummy_signal = "#"
        stop_word_path = "../config/nlp/stop_word_tw.txt"
        dummy_corpus_label = language.create_stop_labels(stop_word_path, dummy_signal, top_k=augment_count)
        samples += [[d[1],d[0]] for d in dummy_corpus_label]
        label_dist = Counter([s[0] for s in samples])
        print (label_dist)
    
    #create label index 
    label_to_idx = {"S":0,"W":1,"O":2,"T":3,"#":4}
    idx_to_label = dict((v, k) for k, v in label_to_idx.items())
    if label_type == "idx" :
        train_samples = [[label_to_idx[d[0]],d[1]] for d in train_samples]
    #
    Y = numpy.array([s[0] for s in train_samples], dtype=object)
    print ("feature engineering")
    X_text=[s[1] for s in train_samples]
    X = embeddings.sentences_to_vectors(X_text)
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
    write_label_feature(X,Y,"../data/{}_{}.csv".format(output_data_file, label_type))
    write_label_feature(X_train,y_train,"../data/{}_{}.csv".format(tran_data_file, label_type))
    write_label_feature(X_test,y_test,"../data/{}_{}.csv".format(test_data_file,label_type))
    
    print("finish")

if __name__ == '__main__' :
    create_label_features("idx",True)