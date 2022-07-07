'''
Created on 2022年6月14日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from text2vec import SBert
from xgboost import XGBClassifier
import mlflow
from mlflow.entities import ViewType
from config import global_config
label_to_idx = global_config.label_to_idx
idx_to_label = global_config.idx_to_label
# S,優勢 (Strengths)
# W,劣勢 (Weaknesses)
# O,機會 (Opportunities)
# T,威脅 (Threats)
label_to_desc = {"S":"優勢 (Strengths)","W":"劣勢 (Weaknesses)","O":"機會 (Opportunities)","T":"威脅 (Threats)","#":"其他 (Others)"}
#
model_xgb = XGBClassifier()
model_path = global_config.best_model_path
model_xgb.load_model(model_path)

#
model_name = global_config.embedding_model_name
embedder = SBert(model_name)
exp_names = [global_config.exp_name_online, global_config.exp_name_offline]

#
import os
 #設定 MinIO 存取
os.environ["MLFLOW_S3_ENDPOINT_URL"] = global_config.MLFLOW_S3_ENDPOINT_URL # 設定S3指向的位置。預設為Amazon S3
os.environ["AWS_ACCESS_KEY_ID"] = global_config.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = global_config.AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = global_config.AWS_DEFAULT_REGION
# 指定了 env 後 s3 才能正確的存取到
model_uri = "s3://mlflow/mlruns/63903915456a4ba4b437ca708a03b8bd/artifacts/model"
def reload_model_by_uri(model_uri):
    model = mlflow.xgboost.load_model(model_uri)
    model_xgb = model
    print ("reload xgb model from:{}".format(model_uri))
    return model_xgb

def reload_model():
    global model_xgb 
    model_path = global_config.best_model_path
    model_xgb.load_model(model_path)
    print ("reload xgb model from:{}".format(model_path))
    return model_xgb

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
    model_name = "swot_ai_xgb"
    test_sentence = ["為了提供更有競爭力的工作團隊","細節上仍有不少出入","政府支持","威脅供給過多"]
    import mlflow.pyfunc

    model_name = "sk-learn-random-forest-reg-model"
    stage = 'Staging'
    
    model = mlflow.pyfunc.load_model(
        model_uri="models:/{}/12".format(model_name)
    )
    
    model.predict(test_sentence[0])

    reload_model_by_uri(model_uri)
    for s in test_sentence:
        print(text_to_swot(s))
    reload_model()
    for s in test_sentence:
        print(text_to_swot(s))
    