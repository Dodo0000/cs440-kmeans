from loader import load_dataset
import random
import numpy as np
import sys

class Cluster:
	def __init__(self, seed):
		self.centroid = seed[:-1]
		self.labels = [0, 0, 0]
		self.centroid_buffer = seed[:-1]
		self.pt_count = 1.
		self.ss_loss = 0.

	def dist(self, point):
		return sum((self.centroid-point[:-1])**2.0) # use squared distance to save time
	
	def clear_points(self):
		self.pt_count = 0.
		self.ss_loss = 0.
		self.labels = [0, 0, 0]
	
	def assign(self, point):
		self.centroid_buffer = self.centroid_buffer * self.pt_count/(self.pt_count+1) + point[:0-1] / (self.pt_count+1) # update centroid position
		self.pt_count += 1
		self.labels[int(point[-1])] += 1
		self.ss_loss += self.dist(point) # add squared distance to loss
	
	def update_centroid(self):
		self.centroid = self.centroid_buffer

if __name__=="__main__":
	assert(len(sys.argv)) == 3
	# load iris dataset
	iris_set = load_dataset()

	# initialize clusters
	seed_examples = random.sample(iris_set, int(sys.argv[1]))
	clusters = [Cluster(seed) for seed in seed_examples]
        
	# iterate
	for i in xrange(int(sys.argv[2])):
		for cluster in clusters:
			cluster.clear_points() # reset points in cluster, keep centroid
		for point in iris_set:
			nearest_cluster = np.argmin([cluster.dist(point) for cluster in clusters])
			clusters[nearest_cluster].assign(point) # reassign points to clusters
		for cluster in clusters:
			cluster.update_centroid() # re-calculate centroids
		if all([c.pt_count==0 for c in clusters]):
			print "removing cluster"
		clusters = [c for c in clusters if c.pt_count > 0] # keep only non-empty clusters
	
	# report error
	for c in clusters:
		print "\nCentroid: "+str(c.centroid)
		print c.labels
	print "SS_total = "+str(sum([c.ss_loss for c in clusters]))
