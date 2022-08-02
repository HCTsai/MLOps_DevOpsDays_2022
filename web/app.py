'''
Created on 2022年6月9日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
# 設定 PYTHONPATH 為專案的根目錄
import sys
from os.path import dirname, abspath
# 設定python path
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)
# 設定 packages
import os
from flask import Flask, jsonify, request, render_template
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge
from prometheus_client import generate_latest
# import logging
from flask import Response
from train import continuous_traning
import time
from threading import Thread
from model_manager import model_inference
from swot import webapi
from config import global_config
from monitoring import model_monitor
from model_manager import model_registry

os.environ["PROMETHEUS_DISABLE_CREATED_SERIES"] = "True"
# logging.basicConfig(filename='log/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# flask app config
app = Flask(__name__)
class CustomFlask(Flask):
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        variable_start_string='%%',  # Default is '{{', I'm changing this because Vue.js uses '{{' / '}}'
        variable_end_string='%%',
    ))
app = CustomFlask(__name__)
app.config['JSON_AS_ASCII'] = False # 讓網頁顯示 JSON中文
# Prometheus Exporter
metrics = PrometheusMetrics(app, path=None)


# metrics for service
predict_counter = Counter("swot_predict_total","number of swot predict")
feature_drift_counter = Counter("swot_feature_drift","number of feature drift")
model_retrain_counter = Counter("swot_model_retrain","number of model re-tranin")
model_metrics = Gauge("swot_model_metrics","metrics for evaluate a ML model (F1, Acc, Recall...)")
model_retrain_time_total = Gauge("swot_model_retrain_time_total","ACC of the model")
uncertain_predict_counter = Counter("swot_model_uncertain_total","uncertain label of the model")
#
exp_name = global_config.exp_name_online

tracking_uri = global_config.local_tracking_uri
if (global_config.mlflow_tracking_type == "1"):
    tracking_uri = global_config.tracking_uri
# 模型的performance 預設值使用線下實驗結果
model_metrics.set(model_registry.get_best_performance(global_config.exp_name_offline, tracking_uri))
# demo 展示用，給定初始資料
predict_counter.set(4328)
feature_drift_counter.set(6)
uncertain_predict_counter.set(2)

def retrain_model_thread(exp_name):
    st = time.time()
    exp_run_ids, best_run_id, perf = continuous_traning.run_pipeline(exp_name)
    et = time.time()
    #update real-time metrics
    model_retrain_counter.inc()
    model_metrics.set(perf)
    model_retrain_time_total.inc(et-st)
    model_inference.reload_model()


@metrics.do_not_track()
@app.route('/model/labelling/list')
def get_labelling_list():
    res = model_monitor.get_label_recommendation()
    return jsonify(res)


@metrics.do_not_track()
@app.route('/model/labelling/<label>/<text>')
def labelling_online(label, text):
    res = {}
    with open("../data/extend_corpus.csv", "a", encoding="utf-8") as f:
        f.write("{},{}\n".format(label, text))
        res["label"] = label 
        res["text"] = text
    return jsonify(res)


@metrics.do_not_track()
@app.route('/model/retrain/<name>')
def retrain(name):
    multi_thread = True
    if name != "" or name is not None:
        exp_name = name
    else:
        exp_name = global_config.exp_name_online
    res = {}
    res["exp_name"] = exp_name
    try:
        if multi_thread:
            t = Thread(target=retrain_model_thread, args=(exp_name,))
            t.start()
        else:
            exp_run_ids, best_run_id, f1 = continuous_traning.run_pipeline(exp_name)
            res["exp_run_ids"] = exp_run_ids
            res["best_run_id"] = best_run_id
            res["f1"] = f1
    except Exception as e: # work on python 3.x
        print('Failed to training: ' + str(e))
    return jsonify(res)


@app.route('/metrics')
@metrics.do_not_track()
def show_metrics():
    return Response(generate_latest(), content_type='text/plain; charset=utf-8')


@app.route('/model/predict/<text>')
@metrics.do_not_track()
def predict(text):
    res_count = model_monitor.input_monitor([text])
    if (res_count > 0):
        feature_drift_counter.inc(res_count)
    res_json = model_inference.text_to_swot(text)
    predict_counter.inc(1)
    # logging.debug("{},{}".format(res_json["label"], text))
    return jsonify(res_json)


@app.route('/uploads', methods=['POST'])
def uploads():
    res, sentence_len = webapi.process_upload(request)
    predict_counter.inc(sentence_len)
    #
    res_count = model_monitor.output_monitor(res)
    if (res_count > 0):
        uncertain_predict_counter.inc(res_count)
    return jsonify(res)


@app.route('/')
def app_root():
    return render_template("swot.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=global_config.app_port, debug=False)
