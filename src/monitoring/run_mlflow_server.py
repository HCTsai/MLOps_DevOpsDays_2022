'''
Created on 2022年6月14日

@author: Hsiao-Chien Tsai(蔡効謙)
'''


import os 
'''
 mlflow server --backend-store-uri sqlite:///../data/mlflow.db   --default-artifact-root="https://locaohost:5005/mlartifacts" -h 127.0.0.1 -p 5005
'''
 
 
if __name__ == '__main__' :
    ip = "127.0.0.1"
    port = 5005 
    command = "mlflow server --backend-store-uri sqlite:///../data/mlflow.db --default-artifact-root= \"https://{}:{}/mlartifacts\" -h {} -p {}".format(ip,port,ip,port)
    conda_path = "D:\python\Anaconda3\envs\bi\python.exe"
    os.system(conda_path + " " + command)