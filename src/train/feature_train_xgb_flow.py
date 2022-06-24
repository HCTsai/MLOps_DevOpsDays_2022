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



# enable autologging
# mlflow.sklearn.autolog()
# mlflow.xgboost.autolog()
# project
tran_data_file = "../data/train_label_features_idx.csv"
test_data_file = "../data/test_label_features_idx.csv"

def get_data():
    print("get data:")
    label_to_idx = {"S": 0, "W": 1, "O": 2, "T": 3, "#": 4}
    idx_to_label = dict((v, k) for k, v in label_to_idx.items())
    
    train_data_list = filetools.file_to_list(tran_data_file)
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
    runs = client.search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"], max_results=1)
    p = 0
    if len(runs) > 0 :
        p = runs[0].data.metrics["acc"]
        print ("預期ACC指標:{}".format(runs[0].data.metrics["acc"]))
    return p
        
def save_best_model(exp_name, exp_ids, model_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(exp.experiment_id, "",run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.acc DESC"], max_results=20)
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
    model = mlflow.xgboost.load_model("runs:/" + runs[0].info.run_id + "/model")
    model_path = "../data/model/xgb_swot_model_best.json"
    model.save_model(model_path)
    return runs[0].info.run_id, runs[0].data.metrics["acc"]

def run_experiment(exp_name, tracking_url, artifact_location):
    # 設定 MLflow 參數
    # 須注意，同一個實驗，指定設定了 artifact_location 就無法修改
    # mlflow.create_experiment(exp_name, artifact_location)
    # mlflow.create_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    # new  experiment
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
    exp_param = [200]
    exp_run_ids = []
    for i, p in enumerate(exp_param):
        with mlflow.start_run() as run:
            print ("model fit, run:{}".format(i))
            print("artifact_uri:{}".format(mlflow.get_artifact_uri()))  # should print out an s3 bucket path
            model_xgb = XGBClassifier(learning_rate = 0.1, n_estimators=p, max_depth=5, min_child_weight=1,
                gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective= 'multi:softmax', nthread=4, seed=27, eval_metric='mlogloss')
            #model fit
            model_xgb.fit(X_train, y_train)
            model_path = "../data/model/xgb_swot_model.json"
            model_xgb.save_model(model_path)
            y_pred = model_xgb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print ("run:{} acc:{}".format(i, accuracy))
            params = model_xgb.get_xgb_params()
            mlflow.log_param("eval_metric", params["eval_metric"])
            mlflow.log_param("objective", params["objective"])
            mlflow.log_param("n_estimators", p)
            mlflow.log_metric("acc", accuracy)
            mlflow.sklearn.eval_and_log_metrics(model_xgb, X_test, y_test, prefix="val_")
            exp_run_ids.append(run.info.run_id)
            # 儲存多次實驗的最好結果
    best_run_id, acc =save_best_model(exp_name, exp_run_ids, model_name)
    return exp_run_ids, best_run_id, acc
if __name__ == '__main__' :
    # mlflow server --backend-store-uri mlflow.db --default-artifact-root s3://mlflow -h 10.56.211.125 -p 5000
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    # os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    # os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    exp_name = "SWOT_AI_離線訓練"
    tracking_uri = "sqlite:///../data/mlflow.db"
    artifact_location = "file:/./mlruns"
    tracking_uri = "mysql+pymysql://mlflow_user:mlflow_user@localhost:3307/mlflow"
    #artifact_location = "http://127.0.0.1:9000/mlflow/mlruns"
    get_best_performance(exp_name,tracking_uri)
    run_experiment(exp_name, tracking_uri, artifact_location)