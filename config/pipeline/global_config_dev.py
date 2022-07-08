'''
Created on 2022年7月6日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

# data
input_file_path = "../data/swot_4000.xlsx"
input_file_path_ext ="../data/extend_corpus.csv"
train_data_list = "../data/train_label_features_idx.csv"
test_data_list = "../data/test_label_features_idx.csv"
label_to_idx = {"S":0,"W":1,"O":2,"T":3,"#":4}
idx_to_label = dict((v, k) for k, v in label_to_idx.items())
# nlp
embedding_model_name = "hfl/chinese-roberta-wwm-ext"
# model drift
feature_drift_model_name = "../data/drift/feature_drift.pkl"
# training 
mlflow_tracking_type = 1 # 0: local tracking 1:remote tracking
# mlflow with local tracking
local_tracking_uri = "sqlite:///../data/mlflow.db"
local_artifact_location = "file:/./mlruns"
# mlflow with remote tracking
cluster_ip = "127.0.0.1"
MLFLOW_S3_ENDPOINT_URL="http://{}:9000".format(cluster_ip) # 設定S3指向的位置。預設為Amazon S3
AWS_ACCESS_KEY_ID = "minioadmin"
AWS_SECRET_ACCESS_KEY = "minioadmin"
AWS_DEFAULT_REGION = ""
tracking_uri = "mysql+pymysql://mlflow_user:mlflow_user@{}:3307/mlflow".format(cluster_ip) 
artifact_location = "s3://mlflow/mlruns/"
# Experiment
exp_name_offline = "swot_exp_offline"
exp_name_online = "swot_exp_online"
best_model_path = "../data/model/xgb_swot_model_best.json"
# web 
app_port = 5000
