'''
Created on 2022年6月14日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
import mlflow 
from mlflow.entities import ViewType


def get_exp_and_run():
    mlflow.set_tracking_uri("sqlite:///../data/mlflow.db")
    exp_count = len(mlflow.list_experiments())
    run_count = 0
    for exp in mlflow.list_experiments() :
        #print(exp.experiment_id)
        run_info_list = mlflow.list_run_infos(exp.experiment_id, run_view_type=ViewType.ALL, order_by=["metric.click_rate DESC"])
        run_count += len(run_info_list)
   

def get_runs_by_exp(exp_name):
    mlflow.set_tracking_uri("sqlite:///../data/mlflow.db")
    exp = mlflow.get_experiment_by_name(exp_name)
    runs = mlflow.tracking.MlflowClient().search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"])
    print (len(runs))
exp_name = "exp_strategy_ai"
get_runs_by_exp(exp_name)

mlflow.set_tracking_uri("sqlite:///../data/mlflow.db")
client = mlflow.tracking.MlflowClient()
exp = mlflow.get_experiment_by_name(exp_name)

run = client.search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"], max_results=1)[0]
print (run.info.run_id)
model = mlflow.sklearn.load_model("runs:/" + run.info.run_id + "/model")
model_path = "../deployment/xgb_swot_model_best.json"
model.save_model(model_path)

mlflow.register_model()

