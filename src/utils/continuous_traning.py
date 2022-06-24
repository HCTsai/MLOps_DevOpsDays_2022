'''
Created on 2022年6月21日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

from features import auto_feature
from train import feature_train_xgb_flow


def traning(exp_name):
    # 測試，先移除特徵積計算
    auto_feature.create_label_features(label_type="idx", label_balance=True)
    
    if exp_name == "" or exp_name == "None" :
        exp_name = "swot_ai_ct"
    tracking_uri = "sqlite:///../data/mlflow.db"
    artifact_location = "file:/./mlruns"
    exp_run_ids, best_run_id, acc = feature_train_xgb_flow.run_experiment(exp_name, tracking_uri, artifact_location)
    return exp_run_ids, best_run_id, acc

if __name__ == '__main__' :
    traning("")