'''
Created on 2022年6月14日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from text2vec import SBert
from xgboost import XGBClassifier
import mlflow
from mlflow.entities import ViewType
label_to_idx = {"S":0,"W":1,"O":2,"T":3,"#":4}
idx_to_label = dict((v, k) for k, v in label_to_idx.items())
# S,優勢 (Strengths)
# W,劣勢 (Weaknesses)
# O,機會 (Opportunities)
# T,威脅 (Threats)
label_to_desc = {"S":"優勢 (Strengths)","W":"劣勢 (Weaknesses)","O":"機會 (Opportunities)","T":"威脅 (Threats)","#":"其他 (Others)"}
#
model_xgb = XGBClassifier()
model_path = "../data/model/xgb_swot_model_best.json"
model_xgb.load_model(model_path)

#
model_name = "hfl/chinese-roberta-wwm-ext"
embedder = SBert(model_name)
exp_names = ["swot_ai_offline", "swot_ai_ct"]

#
model_uri = "http://127.0.0.1:9000/mlflow/mlruns/11c066149d324f1a99688b1589e30f82/artifacts/model"
def reload_model_by_uri():
    
    model = mlflow.xgboost.load_model(model_uri)
    model_xgb = model
    print ("reload xgb model from:{}".format(model_uri))

def reload_model():
    global model_xgb 
    model_path = "../data/model/xgb_swot_model_best.json"
    model_xgb.load_model(model_path)
    print ("reload xgb model from:{}".format(model_path))

def text_to_swot(text):
    
    features = embedder.encode([text])
    label_index = model_xgb.predict(features)[0]
    prob = model_xgb.predict_proba(features)[0][label_index]
    res = {}
    res["text"] = text 
    res["label"] = idx_to_label[label_index]
    res["prob"] = float(prob) #numpy float32 to float
    res["desc"] = label_to_desc[res["label"]] 
    return res

if __name__ == '__main__' :
    test_sentence = ["為了提供更有競爭力的工作團隊","細節上仍有不少出入","政府支持","威脅供給過多"]
   
    reload_model_by_uri()
    for s in test_sentence:
        print(text_to_swot(s))
    reload_model()
    for s in test_sentence:
        print(text_to_swot(s))
    