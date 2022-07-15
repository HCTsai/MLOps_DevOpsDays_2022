# MLOps  

* 保障與監控AI產品的商業價值
* 使AI產品開發成本降低，工作效率提升。
* 線上服務可管理監控，模型品質不佳可監控，可重新訓練。
* 實驗過程自動記錄，資料版本管理，模型版本管理。





### MLOps 活動介紹

<img src="web/static/img/archtecture01.png" width="800">



* 水平整合：商業價值統計，AI使用量，線上資料品質，線上預測品質，模型品質紀錄，模型開發環境紀錄，模型實驗參數紀錄，ML Pipeline程式碼紀錄。
* 垂直整合：AI服務監控，AI模型品質監控，容器環境監控，網路可用性監控，硬體計算資源監控




###如何安裝 MLOps 服務:

建立 MLOps 基礎服務 (MLflow + MySQL + MinIO + Prometheus + Grafana) :
整體架構如下：
<img src="web/static/img/archtecture02.png" width="800">



####安裝步驟 
	docker-compose up

####驗證各種服務正常運作

* MLflow: your_ip:5001
* MySQL:  your_ip:3307
* MinIO:  your_ip:9000
* Prometheus: your_ip:9090
* Grafana: your_ip:3000

docker-compose 的設定文件在 .env，可以改各系統的預設帳號/密碼/Port等設定。

###安裝 AI Service :
#####Run on docker container:

	docker build -t swot_ai:v1 -f "./mlops_docker/dockerfiles/ai_app_service/Dockerfile" .
	docker run --name swot_ai_service -p 5000:5000

#####Run on physical machine :

	pip install requirements.txt
	cd web
	python app.py
	
	瀏覽器開啟： your_ip:5000

 <img src="web/static/img/swot_web_demo_01.png" width="800">


####後台監控畫面 (Grafana) 
預設存取位置: your_ip:3000
預設帳號密碼: admin/admin

<img src="web/static/img/dashboard01.png" width="800">
<img src="web/static/img/dashboard02.png" width="800">


####模型訓練實驗，自動記錄參數，指標，自動產生模型版本控制　(MLflow) 
模型訓練程式碼：
* train/feature_train_xgb_flow.py

執行程式碼後，除了產生模型，也在MLflow中，儲存模型紀錄，與模型版本

<img src="web/static/img/exp_runs.PNG" width="800">

<img src="web/static/img/model_version.PNG" width="800">

####模型儲存倉庫　(MinIO)
不同版本的模型，可設定儲存在　Amazon S3 或是　MinIO:

 <img src="web/static/img/minio_model.PNG" width="800">

####其他注意事項

######使用　MySQL　做為　MLFlow 的　tracking storage
須注意　experiment name 預設不可直接使用中文。

(pymysql.err.OperationalError) (1267, "Illegal mix of collations (latin1_swedish_ci,IMPLICIT) and (utf8mb4_general_ci,COERCIBLE) 
修改方式：

	SET collation_connection = 'utf8_general_ci';

######MLflow 預設使用了　matplotlib 繪製實驗結果　
須注意　MLflow 的實驗跑在其他thread 會導致matplotlib產生錯誤
RuntimeError: main thread is not in main loop


修改方式：(修改 matplotlib backend)

https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask

	import matplotlib
	matplotlib.use("svg")
	import matplotlib.pyplot as plt


