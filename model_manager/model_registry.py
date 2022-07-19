'''
Created on 2022年7月8日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from mlflow.store import artifact
from mlflow.entities import ViewType
import mlflow
from config import global_config
from utils import mlflow_settings
def print_auto_logged_info(r):
    from mlflow.tracking import MlflowClient
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
    
def get_best_performance(exp_name, tracking_url):
    mlflow.set_tracking_uri(tracking_url)
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp :
        return 0
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.val_f1_score DESC"], max_results=1)
    p = 0
    if len(runs) > 0 :
        p = runs[0].data.metrics["val_f1_score"]
    return p
def reload_model_by_uri(model_uri):
    
    model = mlflow.xgboost.load_model(model_uri)
    model_xgb = model
    print ("reload xgb model from:{}".format(model_uri))
    return model_xgb

def get_best_model_runs(exp_name, tracking_url):
    mlflow.set_tracking_uri(tracking_url)
    exp = mlflow.get_experiment_by_name(exp_name)
    #print(exp)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.val_f1_score DESC"], max_results=20)
    return runs
def save_best_model_by_expid(exp_name, exp_ids):
    #取得 同一個實驗 較好的
    runs = get_best_model_runs(exp_name, mlflow_settings.tracking_uri)
    # 註冊模型，只記錄每次實驗，多個run 裡面指標表現最好的模型
    for run in runs:
        run_id = run.info.run_id
        if (run_id in exp_ids) :
            model_uri = "runs:/"+run_id+"/model"
            reg_result = mlflow.register_model(model_uri, mlflow_settings.exp_model_name)
            print(".register new model name:{} version:{}".format(reg_result.name, reg_result.version))
            break 
        
            
    # save model from model registry to local files
    
    # save current_best model 紀錄歷史實驗裡面，指標最好的模型
    print ("best model:{}".format("runs:/" + runs[0].info.run_id + "/model"))
    if mlflow_settings.mlflow_tracking_type == 1 :
        model_uri = "{}/{}/artifacts/model".format(global_config.artifact_location, runs[0].info.run_id)
        model = reload_model_by_uri(model_uri)
        model_path = global_config.best_model_path 
        model.save_model(model_path)
    else :
        model = mlflow.xgboost.load_model("runs:/" + runs[0].info.run_id + "/model")
        model_path = global_config.best_model_path
        model.save_model(model_path)
    return runs[0].info.run_id, runs[0].data.metrics["val_f1_score"]