'''
Created on 2022年6月13日

@author: Hsiao-Chien Tsai(蔡効謙)
'''

import mlflow
from mlflow.entities import ViewType
from mlflow.entities import Run
from datetime import datetime
from prometheus_client import start_http_server
from prometheus_client import Counter, Gauge
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest
from flask import Response

     
tracking_uri = "sqlite:///../data/mlflow.db"
exp_names = ["swot_ai_offline", "swot_ai_ct"]
 
num_of_exp_total = Gauge("mlflow_num_of_exp_total","number of experiments")
num_of_run_total = Gauge("mlflow_num_of_run_total","number of runs")
num_of_run_exp = Gauge("mlflow_num_of_run_exp","number of runs by experiments",['exp_name'])
app = Flask(__name__)
#metrics = PrometheusMetrics(app)
#overwrite default path behavior
metrics = PrometheusMetrics(app,path=None)
app.config['JSON_AS_ASCII'] = False # 讓網頁顯示 JSON中文

def get_exp_and_run():
    mlflow.set_tracking_uri(tracking_uri)
    exp_count = len(mlflow.list_experiments())
    run_count = 0
    for exp in mlflow.list_experiments() :
        #print(exp.experiment_id)
        run_info_list = mlflow.list_run_infos(exp.experiment_id, run_view_type=ViewType.ALL, order_by=["metric.click_rate DESC"])
        run_count += len(run_info_list)
    
    
    num_of_exp_total.set(exp_count)
    num_of_run_total.set(run_count)
    
    
def get_runs_by_exp(exp_names):
    mlflow.set_tracking_uri(tracking_uri)
    run_count = 0
    for exp_name in exp_names :
        exp = mlflow.get_experiment_by_name(exp_name)
        runs = mlflow.tracking.MlflowClient().search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"])
        run_count += len(runs)
    num_of_run_exp.labels(exp_name).set(run_count)
    
@app.route('/metrics')
def metrics():
    get_exp_and_run()
    get_runs_by_exp(exp_names)
    return Response(generate_latest(), content_type='text/plain; charset=utf-8')
    


if __name__ == '__main__' :
    
    app.run(host="0.0.0.0", port=5002, debug=False)