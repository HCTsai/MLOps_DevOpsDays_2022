'''
Created on 2022年6月9日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
# 設定 PYTHONPATH 為專案的根目錄
import sys
from os.path import dirname, abspath
project_root = dirname(dirname(abspath(__file__))) # /h
sys.path.append(project_root)
# 設定 packages 
import os
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge
from prometheus_client import generate_latest
os.environ["PROMETHEUS_DISABLE_CREATED_SERIES"] = "True"

from features import feature_drift_detector
#

from flask import Response
from utils import continuous_traning
import time
from threading import Thread
from train import model_inference 

#flask app config
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 讓網頁顯示 JSON中文
#Prometheus Exporter
metrics = PrometheusMetrics(app, path=None)

#metrics for service
predict_counter = Counter("swot_predict_total","number of swot predict")
feature_drift_counter = Counter("swot_feature_drift","number of feature drift")
model_retrain_counter = Counter("swot_model_retrain","number of model re-tranin")
model_acc = Gauge("swot_model_acc","ACC of the model")
model_retrain_time_total = Gauge("swot_model_retrain_time_total","ACC of the model")



def retrain_model_thread(exp_name):
    st = time.time()
    exp_run_ids, best_run_id, acc = continuous_traning.traning(exp_name)
    et = time.time()
    #update real-time metrics
    model_retrain_counter.inc()
    model_acc.set(acc)
    model_retrain_time_total.inc(et-st)
    model_inference.reload_model()
@metrics.do_not_track()
@app.route('/model/labelling/<label>/<text>')
def labelling_online(label, text):
    res = {}
    with open("../data/extend_corpus.csv","a",encoding="utf-8") as f:
        f.write("{},{}\n".format(label,text))
        res["label"] = label 
        res["text"] = text
    return res 
@metrics.do_not_track()
@app.route('/model/retrain')
@app.route('/model/retrain/<name>')
def retrain(name):
    #
    exp_name = "swot_ai_ct"
    if name != "" or name != None :
        exp_name=name
    res = {}
    res["exp_name"] = exp_name;
    try :
        t = Thread(target=retrain_model_thread, args=(exp_name,))
        t.start()
        #exp_run_ids, best_run_id, acc = continuous_traning.traning(exp_name)
        #res["exp_run_ids"] = exp_run_ids
        #res["best_run_id"] = best_run_id
        #res["acc"] = acc
        
    except Exception as e: # work on python 3.x
        print('Failed to traning: '+ str(e))
    #exp_run_ids, best_run_id, acc = continuous_traning.traning(exp_name)
    #To-Do re-tranin time elapse \
    return res

@app.route('/metrics')
@metrics.do_not_track()
def show_metrics():
    return Response(generate_latest(), content_type='text/plain; charset=utf-8')



@app.route('/model/predict/<text>')
@metrics.do_not_track()
def predict(text):
    predict_counter.inc(1)
    res_json = model_inference.text_to_swot(text)
    
    #feature drift detection
    res = feature_drift_detector.detect_feature_drift([text])
    if (len(res)>0) :
        print ("feature drift:")
        print(res)
        feature_drift_counter.inc(1)
    
    return res_json

if __name__ == '__main__' :
    
    app.run(host="0.0.0.0", port=5000, debug=False)