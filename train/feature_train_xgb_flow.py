'''
Created on 2022年5月4日

@author: Hsiao-Chien Tsai
'''
# 讀取 Labeling data 產生 feature and label
# label:{0,1,2,3}  #features 768 dimensions
import mlflow

from utils import filetools
from text2vec import SBert
import numpy as np
#

from mlflow.entities import ViewType
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from nlp import language
import os 
from mlflow.store import artifact
from config import global_config


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
# functions
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
        print ("預期 F-score:{}".format(runs[0].data.metrics["val_f1_score"]))
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
def save_best_model(exp_name, exp_ids, model_name, tracking_type, tracking_url):
    #取得 同一個實驗 較好的
    runs = get_best_model_runs(exp_name, tracking_url)
    # 註冊模型，只記錄每次實驗，多個run 裡面指標表現最好的模型
    for run in runs:
        run_id = run.info.run_id
        if (run_id in exp_ids) :
            model_uri = "runs:/"+run_id+"/model"
            reg_result = mlflow.register_model(model_uri, model_name)
            print(".register model name:{} version:{}".format(reg_result.name, reg_result.version))
            break 
    #save current_best model 紀錄歷史實驗裡面，指標最好的模型
    print ("best model:{}".format("runs:/" + runs[0].info.run_id + "/model"))
    
    if tracking_type == 1 :
        model_uri = "s3://mlflow/mlruns/{}/artifacts/model".format(runs[0].info.run_id)
        model = reload_model_by_uri(model_uri)
        model_path = global_config.best_model_path 
        model.save_model(model_path)
        
    else :
        model = mlflow.xgboost.load_model("runs:/" + runs[0].info.run_id + "/model")
        model_path = "../data/model/xgb_swot_model_best.json"
        model.save_model(model_path)
    
    
    return runs[0].info.run_id, runs[0].data.metrics["val_f1_score"]

def run_experiment(exp_name, tracking_url, artifact_location, tracking_type):
    # 設定 MLflow 參數
    # 須注意，同一個實驗，指定設定了 artifact_location 就無法修改
    # mlflow.create_experiment(exp_name, artifact_location)
    # mlflow.create_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    # new  experiment , exp 產生時就要決定好 artifact_location
    if not exp :
        mlflow.create_experiment(exp_name, artifact_location)
    
    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment(exp_name)
    # autolog 可設定各種紀錄參數
    mlflow.xgboost.autolog()
    model_name = "swot_ai_xgb"
    
    #
    X_train, X_test, y_train, y_test = get_data()
    #設定模型參數
    exp_param = [100]
    exp_run_ids = []
    for i, p in enumerate(exp_param):
        with mlflow.start_run() as run:
            print ("model fit, run:{}".format(i))
            print("artifact_uri:{}".format(mlflow.get_artifact_uri()))  # should print out an s3 bucket path
           
            model_xgb = XGBClassifier(learning_rate = 0.05, n_estimators=p, max_depth=5, min_child_weight=1,
                gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective= 'multi:softmax', nthread=4, seed=27, eval_metric='mlogloss')
            #model fit
            model_xgb.fit(X_train, y_train)
            model_path = "../data/model/xgb_swot_model.json"
            model_xgb.save_model(model_path)
            y_pred = model_xgb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print ("run:{} acc:{}".format(i, accuracy))
            #logging
            params = model_xgb.get_xgb_params()
            mlflow.log_param("eval_metric", params["eval_metric"])
            mlflow.log_param("objective", params["objective"])
            mlflow.log_param("n_estimators", p)
            mlflow.log_metric("acc", accuracy)
            mlflow.set_tag("artifact-root", mlflow.get_artifact_uri())
            mlflow.sklearn.eval_and_log_metrics(model_xgb, X_test, y_test, prefix="val_")
            exp_run_ids.append(run.info.run_id)
            # 儲存多次實驗的最好結果
    best_run_id = ""
    model_metrics = 0 
    try :
        best_run_id, model_metrics = save_best_model(exp_name, exp_run_ids, model_name, tracking_type, tracking_url)
    except Exception as e: 
        print('Failed to save_best_model: '+ str(e))
    return exp_run_ids, best_run_id, model_metrics
if __name__ == '__main__' :
    
    
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
    
    exp_name = global_config.exp_name_offline
    get_best_performance(exp_name, tracking_uri)
    run_experiment(exp_name, tracking_uri, artifact_location, mlflow_tracking_type)