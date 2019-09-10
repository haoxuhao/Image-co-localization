#encoding=utf-8
import numpy as np
# from Bio.Cluster import *
# from numpy import linalg as LA
import mpl_toolkits.axisartist as axisartist
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from spherecluster import SphericalKMeans

class SphereCluster(object):
    def __init__(self, num_cluster=2):
        self.name = "SphereCluster"
        self.ncluster=num_cluster
        self.clusters = []
        self.main_id = 0
        self.mean = None
    
    def l2norm(self, data, axies=0, mean=None):
        process_data = normalize(data, norm="l2")
        return process_data

    def fit(self, data, norm=False):
        '''
        Args:
            data numpy.ndarray: [m, n] m samples every sample with n dimention
            norm boolean: False as default, 
        '''
        if not norm:
            self.mean = np.mean(data, axis=0)
            centered = data-self.mean
            process_data = self.l2norm(centered)
        else:
            process_data = data
        #clusterid, error, nfound = kcluster(process_data, nclusters=self.ncluster)#dist="u"
        #cdata, cmask = clustercentroids(process_data, mask=None, transpose=0, clusterid=clusterid, method='a')
        skm = SphericalKMeans(n_clusters=self.ncluster, verbose=0)
        skm.fit(process_data)
        self.clusters=skm.cluster_centers_
        self.clusterid = skm.labels_
        self.loss = skm.inertia_
        scores = []
        for i in range(self.ncluster):
            idxs = np.where(self.clusterid==i)[0].tolist()
            cluster_data = process_data[idxs,:]
            confs = np.dot(cluster_data, self.clusters[i,:].T)
            #print(confs)
            score = np.mean(confs)
            scores.append(score)
        #print(scores)
        self.main_id = np.argmin(scores)
        print(self.main_id)

        return self.clusters, self.clusterid, self.main_id

    def predict(self, data, norm=False):
        '''
        Args:
            data numpy.ndarray: shape [n, c]
        return:
            labeled numpy.ndarray: shape [n, ncluster]
        '''
        data -= self.mean
        process_data = self.l2norm(data)

        #label all vectors
        dist_matrix = (np.dot(process_data, self.clusters.T))
        labeled = np.argmax(dist_matrix, axis=1)

        return labeled


if __name__ == "__main__":
    data = np.array(range(2*512*4*4),dtype=np.float32)
    data = data.reshape(2,512,4,4)
    
    SC = SphereCluster()
    flatend_data = data.transpose(1,0,2,3)
    flatend_data = flatend_data.reshape(-1, flatend_data.shape[0])
    
    SC.fit(flatend_data)
    SC.predict(data)
