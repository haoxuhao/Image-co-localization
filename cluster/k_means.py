#encoding=utf-8

#encoding=utf-8

import numpy as np
from Bio.Cluster import *
from numpy import linalg as LA
import mpl_toolkits.axisartist as axisartist
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class KCluster(object):
    def __init__(self, num_cluster=2):
        self.name = "SphereCluster"
        self.ncluster=num_cluster
        self.clusters = []
        self.main_id = 0
        self.mean = None
    
    def l2norm(self, data, axies=0, mean=None):
        process_data = normalize(data, norm="l2")
        return data
        # return process_data

    def fit(self, data, norm=False):
        '''
        Args:
            data numpy.ndarray: [m, n] m samples every sample with n dimention
            norm boolean: False as default, 
        '''
        if not norm:
            self.mean = np.mean(data, axis=0)
            centered = data-self.mean
            co_var_matrix = (1.0/centered.shape[0])*np.dot(centered.T, centered)
            np.save("../data/tmp/co_var_matrix.npy", co_var_matrix)
            process_data = self.l2norm(centered)
        else:
            process_data = data
        clusterid, error, nfound = kcluster(process_data, nclusters=self.ncluster)#dist="u"
        cdata, cmask = clustercentroids(process_data, mask=None, transpose=0, clusterid=clusterid, method='a')
        result = {}
        scores = []
        self.clusters = []
        self.labes = []
        for i in range(self.ncluster):
            label = np.where(clusterid==i)[0].tolist()
            self.labes.append(label)
            cluster_data = process_data[label,:]
            cluster = np.mean(cluster_data, axis=0)
            #cluster = normalize(cluster.reshape(1,cluster.shape[0]), norm='l2')
            result[i]=cluster
            self.clusters.append(cluster)
            
            #score = np.mean(np.dot(cluster_data, cluster.T))
            scores.append(len(label))
        print(scores)
        self.clusters = np.vstack(self.clusters)
        self.main_id = np.argmax(scores)

        return result, clusterid, self.main_id

    def predict(self, data):
        '''
        Args:
            data numpy.ndarray: shape [n, c, h, w]
        return:
            labeled numpy.ndarray: shape [n, 1, h, w]
        '''
        #reshape [n,c,h,w]->[n*h*w, c]
        n, c, h, w = data.shape
        process_data = data.transpose(1,0,2,3)
        process_data = process_data.reshape(-1, process_data.shape[0])
        process_data -= self.mean
        #process_data = self.l2norm(process_data)

        #label all vectors
        dist_matrix = (np.dot(process_data,self.clusters.T))

        #dist_matrix = np.square()

        labeled = np.argmax(dist_matrix, axis=1)

        # reshape to origin [n*h*w, c]->[n, h, w]
        labeled = labeled.reshape(n, h, w)

        return labeled


if __name__ == "__main__":
    data = np.array(range(2*512*4*4),dtype=np.float32)
    data = data.reshape(2,512,4,4)
    flatend_data = data.transpose(1,0,2,3)
    flatend_data = flatend_data.reshape(-1, flatend_data.shape[0])

    KC = KCluster()
    KC.fit(flatend_data)
    # SC = SphereCluster()
    
    
    # SC.fit(flatend_data)
    # SC.predict(data)
