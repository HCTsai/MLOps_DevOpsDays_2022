'''
Created on 2022年6月20日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
import mlflow

expr_name = "new_experiment111"  # create a new experiment (do not replace)
s3_bucket = "http://127.0.0.1:9000/mlflow"  # replace this value
mlflow.create_experiment(expr_name, s3_bucket)
mlflow.set_experiment(expr_name)

with mlflow.start_run():
    print(mlflow.get_artifact_uri())  # should print out an s3 bucket path
    
    # create a file to log
    with open("test.txt", "w") as f:
        f.write("test")

    mlflow.log_artifact("test.txt")