'''
Created on 2022年6月21日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

from features import auto_feature
from train import feature_train_xgb_flow
from config import global_config
import os

def traning(exp_name):
    # 測試，先移除特徵積計算
    auto_feature.create_label_features(label_type="idx", label_balance=True)
    
    mlflow_tracking_type = global_config.mlflow_tracking_type
    tracking_uri = global_config.local_tracking_uri
    artifact_location = global_config.local_artifact_location
    if mlflow_tracking_type == 1 : #  MySQL + MinIO
        #設定 MinIO 存取
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = global_config.MLFLOW_S3_ENDPOINT_URL # 設定S3指向的位置。預設為Amazon S3
        os.environ["AWS_ACCESS_KEY_ID"] = global_config.AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = global_config.AWS_SECRET_ACCESS_KEY
        os.environ["AWS_DEFAULT_REGION"] = global_config.AWS_DEFAULT_REGION
        tracking_uri = global_config.tracking_uri 
        artifact_location = global_config.artifact_location
    
    if exp_name == "" or exp_name == "None" :
        exp_name = "swot_ai_ct"
    exp_run_ids, best_run_id, model_metrics = feature_train_xgb_flow.run_experiment(exp_name, tracking_uri, artifact_location)
    return exp_run_ids, best_run_id, model_metrics

if __name__ == '__main__' :
    traning("")