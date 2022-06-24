'''
Created on 2022年6月15日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
from nlp import embeddings

feature_drift_model = {}
'''
feature_drift_model["clusters"] = centroid_means 
feature_drift_model["drift_dist_threshold"] = drift_dist_threshold
feature_drift_model["drift_dist_function"] = distance_to_clusters
feature_drift_model["sample_size"] = len(X_train)
'''


def get_vector_mean(vlist):
    return np.mean(vlist, axis=0)

def cos_distance(v1, v2):
    cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
    return cos_sim

def distance_point_list(point, vlist):
    res = [] 
    for v in vlist:
        res.append(cos_distance(point, v))
    return res
def distance_to_clusters(cluster_list, vlist):
    res = [] 
    for v in vlist:
        max_score = 0
        for c in cluster_list:
            score = cos_distance(c, v)
            if score > max_score:
                max_score = score
        res.append(max_score)
    return res

def load_feature_drift_model(model_path):
    global feature_drift_model 
    with open(model_path,"rb") as f:
        feature_drift_model = pickle.load(f)
        
    #print (feature_drift_model)
def detect_feature_drift(text_list, drift_dist_threshold=None):
    vectors = embeddings.sentences_to_vectors(text_list)
    centroid_means = feature_drift_model["clusters"]
    if drift_dist_threshold == None:
        drift_dist_threshold = feature_drift_model["drift_dist_threshold"]
    drift_distances = distance_to_clusters(centroid_means, vectors)
    #print(drift_distances)
    results = []
    for i, d in enumerate(drift_distances) :
        if d < drift_dist_threshold :  
            results.append(text_list[i])
    
    return results

feature_drift_model_name = "../data/drift/feature_drift.pkl"
load_feature_drift_model(feature_drift_model_name)

if __name__ == '__main__':
    test_list = ["a這依據是非 化沒有特別ˋ意義'","今日我們需發展專業","面對外在的競爭","謝謝","aaaaaaaaaaaaa"]
    res = detect_feature_drift(test_list)
    for i, text in enumerate(res) :
        print("drift {}: {}".format(i, text))
    
    