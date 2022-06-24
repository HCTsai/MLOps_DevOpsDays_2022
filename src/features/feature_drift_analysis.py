'''
Created on 2022年6月15日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from utils import filetools
from nlp.language import sentences
from text2vec import SBert
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
#
import pickle
from nlp import embeddings


#
label_file_path = "../data/swot_4000.xlsx"
data_list = filetools.excel_to_list(label_file_path)
label_idx = 0
feature_idx = [1]
samples = data_list[0:4000]
sentences = [s[1] for s in samples]

train_data_list = filetools.file_to_list("../data/train_label_features_idx.csv")
test_data_list = filetools.file_to_list("../data/test_label_features_idx.csv")
print("test sample {}-->{}".format(test_data_list[0][0], test_data_list[0][1:6]))
#string to float
X_train = np.array([d[1:] for d in train_data_list]).astype(float)
X_test = np.array([d[1:] for d in test_data_list]).astype(float)
print (X_train.shape)
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
        max_score =0
        for c in cluster_list :
            score = cos_distance(c, v)
            if score > max_score:
                max_score = score
        res.append(max_score)
    return res


#取得 Quality 好的 keyword_sentences
keyword_sentences = [s[0] for s in filetools.file_to_list("../config/nlp/SWOT.txt")]
train_means = get_vector_mean(X_train)
test_means = get_vector_mean(X_test)
#centroid_means = get_vector_mean(embeddings.sentences_to_vectors(keyword_sentences))
centroid_cluster = embeddings.sentences_to_vectors(keyword_sentences)

print (centroid_cluster.shape)

#train_distance = distance_point_list(train_means, X_train)
#test_distance = distance_point_list(train_means, X_test)
#sample_embs = embedder.encode(sample_sentences)
#print (distance_point_list(train_means, sample_embs))

train_distance = distance_to_clusters(centroid_cluster, X_train)
test_distance = distance_to_clusters(centroid_cluster, X_test)


bin_size = 100
dist_bin = np.arange(0, bin_size+1)/bin_size # [0,1]
print (dist_bin[0:5])
distribute, bins = np.histogram(a=train_distance, bins=dist_bin)
percentile = distribute/len(train_means)

drift_dist_threshold = 0 # 以 centroid_cluster 為基準，語意語意越接近 centroid 代表有意義的樣本，當相似度低於此值，視為異常值。 
drift_thres_ratio = 0.05 # 10% ﹑設定一個異常值的比例

for i, p in enumerate(list(percentile)) : 
    if p > drift_thres_ratio:
        drift_dist_threshold = bins[i]
        break

print (drift_dist_threshold)
''''
for s in sentences :
    if distance_point_list_v2(centroid_cluster,  embedder.encode([s]))[0] < thres :
        print (s)
'''
   
feature_drift_model = {}
feature_drift_model["clusters"] = centroid_cluster 
feature_drift_model["drift_dist_threshold"] = drift_dist_threshold
#feature_drift_model["drift_dist_function"] = distance_to_clusters
feature_drift_model["sample_size"] = len(X_train)

feature_drift_model_name = "../data/drift/feature_drift.pkl"
with open(feature_drift_model_name,"wb") as f:
    pickle.dump(feature_drift_model, f)
    
with open(feature_drift_model_name,"rb") as f:
    feature_drift_model = pickle.load(f)
#

print("")
kwargs = dict(alpha=0.5, bins=100)
plt.hist(train_distance, **kwargs, color='g', label='Train data')
plt.hist(test_distance, **kwargs, color='b', label='Test data')
plt.gca().set(title='Sentences - Mean distances', ylabel='Sample Count')
plt.legend()
plt.axvline(drift_dist_threshold.mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig("../data/drift/feature_drift_distribution.png")
plt.show()

