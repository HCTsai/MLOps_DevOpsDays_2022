'''
Created on 2022年7月7日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

from features import feature_drift_detector

def input_monitor(text_list):
    res = feature_drift_detector.detect_feature_drift(text_list)
    if (len(res)>0) :
        with open("../data/feature_drift_corpus.csv","a",encoding="utf-8") as f:
            for drift_text in res :
                f.write("{}\n".format(drift_text))
        print ("feature drift:{}".format(res))
    return len(res)

def output_monitor(model_res):
    res_count = 0
    with open("../data/uncertain_corpus.csv","a",encoding="utf-8") as f:
        for e in model_res["table"]:
            if (e["score"] < 0.6):
                text = "{},{},{}\n".format(e["label"], e["text"], e["score"])
                f.write(text)
                res_count += 1
                print ("uncertain label:{}".format(text))
    return res_count


def get_label_recommendation():
    res = {}
    res["uncertain_samples"] = []
    res["feature_drift_samples"] = []
    
    with open("../data/uncertain_corpus.csv", "r",encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        res["uncertain_samples"] = lines[-3:]
    
    with open("../data/feature_drift_corpus.csv", "r",encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        res["feature_drift_samples"] = lines[-3:]
    return res 