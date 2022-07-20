'''
Created on 2022年5月4日

@author: Hsiao-Chien Tsai
'''
# 讀取 Labeling data 產生 feature and label
# label:{0,1,2,3}  #features 768 dimensions=

from utils import filetools
from text2vec import SBert
import numpy as np
#

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from nlp import language
import os 

from config import global_config
from model_manager import model_registry
from mlflow import get_tracking_uri
import mlflow
# enable autologging
# mlflow.sklearn.autolog()
# mlflow.xgboost.autolog()
# project

train_data_file = global_config.train_data_list
test_data_file = global_config.test_data_list

def get_data():
    print("get data:")
    label_to_idx = {"S": 0, "W": 1, "O": 2, "T": 3, "#": 4}
    idx_to_label = dict((v, k) for k, v in label_to_idx.items())
    
    train_data_list = filetools.file_to_list(train_data_file)
    test_data_list = filetools.file_to_list(test_data_file)
    #print("test sample {}-->{}".format(test_data_list[0][0], test_data_list[0][1:6]))
    X_train = np.array([d[1:]for d in train_data_list]).astype(float)
    y_train = np.array([int(d[0]) for d in train_data_list])
    X_test = np.array([d[1:] for d in test_data_list]).astype(float)
    y_test = np.array([int(d[0]) for d in test_data_list])
    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_test, y_train,  y_test

def run_experiment(exp_name, mlflow_tracking_type, tracking_uri, artifact_location):
    """
    run an ML experiment 
    :param str exp_name: experiment name
    :return: experiment results
    """
    print ("run experiment:{}".format(exp_name))
    # 取得訓練資料
    X_train, X_test, y_train, y_test = get_data()
    # 設定模型實驗參數
    param_estimators = [100]
    
    # 設定實驗環境
    exp_run_ids = [] # 紀錄實驗id
    mlflow.set_tracking_uri(tracking_uri) # 設定實驗記錄服務位置
    exp = mlflow.get_experiment_by_name(exp_name) #檢查是否需要產生新的實驗
    if not exp :
        mlflow.create_experiment(exp_name, artifact_location)
        print("create {} on {}".format(exp_name, artifact_location))
    mlflow.set_experiment(exp_name)
    mlflow.xgboost.autolog()
    # 開始做ML實驗
    with mlflow.start_run() as run:
        print("artifact_uri:{}".format(mlflow.get_artifact_uri()))
        print("tracking_uri:{}".format(mlflow.get_tracking_uri()))
        model_xgb = XGBClassifier(
            learning_rate=0.05, n_estimators=param_estimators[0],
            max_depth=4, min_child_weight=1,
            gamma=0, subsample=0.8, colsample_bytree=0.8,
            objective='multi:softmax', nthread=4,
            seed=27, eval_metric='mlogloss')
        # model fit
        model_xgb.fit(X_train, y_train)
        model_path = "../data/model/xgb_swot_model.json"
        model_xgb.save_model(model_path)
        
        # 紀錄自定義的實驗結果
        params = model_xgb.get_xgb_params()
        mlflow.log_param("objective", params["objective"])
        mlflow.set_tag("artifact-root", mlflow.get_artifact_uri())
        # 自動計算模型表現指標，並記錄
        mlflow.sklearn.eval_and_log_metrics(model_xgb, X_test, y_test, prefix="val_")
        exp_run_ids.append(run.info.run_id)
    # 註冊模型，到模型版本庫
    best_run_id = ""
    model_metrics = 0
    best_run_id, model_metrics = model_registry.save_best_model_by_expid(exp_name, exp_run_ids, global_config.exp_model_name, mlflow_tracking_type, tracking_uri)
    return exp_run_ids, best_run_id, model_metrics

if __name__ == '__main__' :
    exp_name = global_config.exp_name_offline
    # 預設使用 local端儲存 mlflow 資料
    tracking_uri = global_config.local_tracking_uri
    artifact_location = global_config.local_artifact_location
    
    mlflow_tracking_type = global_config.mlflow_tracking_type
    if mlflow_tracking_type == 1 : #  MySQL + MinIO
        #設定 MinIO 存取
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = global_config.MLFLOW_S3_ENDPOINT_URL # 設定S3指向的位置。預設為Amazon S3
        os.environ["AWS_ACCESS_KEY_ID"] = global_config.AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = global_config.AWS_SECRET_ACCESS_KEY
        os.environ["AWS_DEFAULT_REGION"] = global_config.AWS_DEFAULT_REGION
        tracking_uri = global_config.tracking_uri 
        artifact_location = global_config.artifact_location

    p = model_registry.get_best_performance(exp_name, tracking_uri)
    print ("預期 F-score:{}".format(p))
    run_experiment(exp_name, mlflow_tracking_type, tracking_uri, artifact_location)
