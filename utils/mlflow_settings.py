'''
Created on 2022年7月15日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from config import global_config
import os
import mlflow

from mlflow.entities import ViewType
from mlflow.store import artifact

# 預設使用 local端儲存 mlflow 資料
tracking_uri = global_config.local_tracking_uri
artifact_location = global_config.local_artifact_location

mlflow_tracking_type = global_config.mlflow_tracking_type
if mlflow_tracking_type == 1 : #  MySQL + MinIO
    #設定 MinIO 存取
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = global_config.MLFLOW_S3_ENDPOINT_URL # 設定S3指向的位置。預設為Amazon S3
    os.environ["AWS_ACCESS_KEY_ID"] = global_config.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = global_config.AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = global_config.AWS_DEFAULT_REGION
    tracking_uri = global_config.tracking_uri 
    artifact_location = global_config.artifact_location
