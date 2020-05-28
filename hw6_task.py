from sklearn.cluster import KMeans
import numpy as np
import sys
import time

def clustering(kmeans, data, k):
	clusters = {}
	for index,i in enumerate(kmeans.labels_):
		if i not in clusters:
			clusters[i] = []
		clusters[i].append(data[index])
	return clusters


def find_comp(clusters):
	retained, compressed, compress_set = [], {}, {}
	for key,val in clusters.items():
		if len(val) == 1:
			retained.append(val[0])
			clusters[key]=-100

	for key,val in clusters.items():
		if val!=-100:
			for i in val:
				if key in compress_set:
					compressed = evaluate(compressed, key, i, 2)
				else:
					compressed = evaluate(compressed, key, i, 1)
					compress_set[key] = []
				compress_set[key].append(i[0])

	return retained, compressed, compress_set

def mahalanobis(s, point, case=1):
	dict1 = {}
	sort_keys=sorted(s.keys())
	for key in sort_keys:
		mean, standard_dev, val = [], [], s[key]
		for index,i in enumerate(val[1]):
			mean.append(i/val[0])
			standard_dev.append((val[2][index]/val[0] - (i/val[0])**2)**0.5)

		l=[]
		for i,v in enumerate(point[2]):
			l.append(((v-mean[i])/standard_dev[i])**2)
		dict1[key] = sum(l)**0.5

	distance = min(dict1.values())
	for key,val in dict1.items():
		if val==distance:
			cluster, distance = key, val
			break

	threshold = 2*((len(point[2]))**0.5)

	if case==2 or distance<threshold:
		return cluster

	return -1


def bfr(discarded, compressed, retained, data, discard_set, compress_set):
	prob_rs = []

	for i in data:
		pred=mahalanobis(discarded, i)
		if pred == -1:
			pred2 = mahalanobis(compressed, i)
			if pred2 == -1:
				prob_rs.append(i)
			else:
				compressed = evaluate(compressed, pred2, i, 2)
				compress_set[pred2].append(i[0])
		else:
			discarded = evaluate(discarded, pred, i, 2)
			discard_set[pred].append(i[0])		


	if len(prob_rs) > 0:
		retained = np.concatenate((retained, np.array(prob_rs)), axis=0)

	no_of_clusters=int(len(retained)*0.9)
	kmeans = KMeans(n_clusters=no_of_clusters, random_state=0).fit(np.ndarray.tolist(retained[:, 2]))
	clusters = clustering(kmeans, retained, no_of_clusters)

	retained, prob_c, prob_new = find_comp(clusters)

	if len(prob_c) != 0:
		for key,val in prob_c.items():
			index = max(sorted(compressed.keys())) +1
			point = [0, 0, [i/val[0] for i in val[1]]]
			label = mahalanobis(compressed, point)
			if label == -1:
				compressed[index+key] = val
				compress_set[index+key] = [p for p in prob_new[key]]
			else:
				zero, one, two=compressed[label][0]+val[0], [], []
				for i, v in enumerate(val[1]):
					one.append(compressed[label][1][i]+v)
				for i, v in enumerate(val[2]):
					two.append(compressed[label][2][i]+v)
				compressed[label]=[zero, one, two]
				for i in val:
					compress_set[label].append(i)

	return discarded, compressed, retained, discard_set, compress_set

def evaluate(s, label, p, case):
	if case==1:
		s[label] = [1, p[2], [d**2 for d in p[2]]]

	if case==2:
		zero, one, two=s[label][0], [], []
		for i , val in enumerate(p[2]):
			one.append(s[label][1][i] + val)
			two.append(s[label][2][i] + val**2)
		s[label]=[zero+1, one, two]
	return s

initial_time = time.time()
input_file = sys.argv[1]
k = int(sys.argv[2])
output_file = sys.argv[3]

file = open(input_file, "r")
points = []
for line in file:
	l = line.replace("\n", "").split(",")
	points.append([int(l[0]), int(l[1]), [float(d) for d in l[2:]]])
file.close()
data = np.array(points)


discard_set = {}
dataset = data[:int(len(data)*0.2), :]
kmeans = KMeans(n_clusters=k*10, random_state=0).fit(np.ndarray.tolist(dataset[:, 2]))
clusters = clustering(kmeans, dataset, k*10)
cluster_set = []
retained = []
for key,val in clusters.items():
	if len(val) == 1:
		retained.append(val[0])
	else:
		for i in val:
			cluster_set.append(i)

cluster_set = np.array(cluster_set)
kmeans = KMeans(n_clusters=k, random_state=0).fit(np.ndarray.tolist(cluster_set[:, 2]))

discarded = {}

for i,val in enumerate(cluster_set):
	if kmeans.labels_[i] in discard_set:
		discarded = evaluate(discarded, kmeans.labels_[i], val, 2)
	else:
		discarded = evaluate(discarded, kmeans.labels_[i], val, 1)
		discard_set[kmeans.labels_[i]] = []
	discard_set[kmeans.labels_[i]].append(val[0])
retained =np.array(retained)


if (len(retained)) == 1 or (len(retained)) == 0:
	compressed, compress_set={}, {}
else:
	kmeans = KMeans(n_clusters=int(len(retained)*0.9), random_state=0).fit(np.ndarray.tolist(retained[:, 2]))
	clusters = clustering(kmeans, retained, int(len(retained)*0.9))
	retained, compressed, compress_set = find_comp(clusters)

f = open(output_file, "w+")
f.write("The intermediate results:\n")
f.write("Round 1: " + str(sum([discarded[key][0] for key in discarded])) + "," + str(len(compressed)) + "," + str(sum([compressed[key][0] for key in compressed])) + "," + str(len(retained)))
f.write("\n")

start = 0.2
end = 0.4

for i in range(4):
	discarded, compressed, retained, discard_set, compress_set = bfr(discarded, compressed, retained, data[int(len(data)*start):int(len(data)*end)], discard_set, compress_set)

	f.write("Round " + str(i+2) + ": " + str(sum([discarded[key][0] for key in discarded])) + "," + str(len(compressed)) + "," + str(sum([compressed[key][0] for key in compressed])) + "," + str(len(retained)))
	f.write("\n")
	start = end
	end += 0.2

for key,val in compressed.items():
	point = [0, 0, [i/val[0] for i in val[1]]]
	label = mahalanobis(discarded, point, 2)
	zero, one, two=discarded[label][0]+val[0], [], []
	for i, v in enumerate(val[1]):
		one.append(discarded[label][1][i]+v)
	for i, v in enumerate(val[2]):
		two.append(discarded[label][2][i]+v)
	discarded[label]=[zero, one, two]
	for p in compress_set[key]:
		discard_set[label].append(p)

f.write("Round 6" + ": " + str(sum([discarded[key][0] for key in discarded])) + ",0,0," + str(len(retained)))
f.write("\n\nThe clustering results:\n")

ans = []
for key,val in discard_set.items():
	for i in val:
		ans.append((i, key))
for i in retained:
	ans.append((i[0],-1))
ans = sorted(ans, key=lambda x: x[0])
for i in ans:
	f.write(str(i[0]) + "," + str(i[1]))
	f.write("\n")

f.close()

final_time = time.time()
print("Duration================================", final_time-initial_time)