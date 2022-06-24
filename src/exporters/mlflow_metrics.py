'''
Created on 2022年6月24日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

import mlflow
from mlflow.entities import ViewType
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge
from prometheus_client import generate_latest

tracking_uri = "sqlite:///../data/mlflow.db"
exp_names = ["swot_ai_offline", "swot_ai_ct"]



def get_exp_and_run():
    mlflow.set_tracking_uri(tracking_uri)
    exp_count = len(mlflow.list_experiments())
    run_count = 0
    for exp in mlflow.list_experiments() :
        #print(exp.experiment_id)
        run_info_list = mlflow.list_run_infos(exp.experiment_id, run_view_type=ViewType.ALL, order_by=["metric.click_rate DESC"])
        run_count += len(run_info_list)
    
    
    return exp_count, run_count
    
    
def get_runs_by_exp(exp_names):
    mlflow.set_tracking_uri(tracking_uri)
    run_count = 0
    for exp_name in exp_names :
        exp = mlflow.get_experiment_by_name(exp_name)
        runs = mlflow.tracking.MlflowClient().search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"])
        run_count += len(runs)
    
    return  run_count