'''
Created on 2022年6月21日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

from features import auto_feature
from train import feature_train_xgb_flow
from utils import mlflow_settings

def run_pipeline(exp_name):
    # 測試，先移除特徵積計算
    auto_feature.create_label_features(label_type="idx", label_balance=True)
    if exp_name == "" or exp_name == "None":
        exp_name = "swot_exp_online"
    mlflow_settings.set_exp_name(exp_name)
    exp_run_ids, best_run_id, model_metrics = feature_train_xgb_flow.run_experiment(exp_name)
    return exp_run_ids, best_run_id, model_metrics

if __name__ == '__main__' :
    run_pipeline("")