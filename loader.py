import csv
import numpy as np

def load_dataset():
  dataset = []
  with open('iris-dataset.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      if len(row) == 0:
        continue
      label = (row[-1] == "Iris-setosa") * 2 + (row[-1] == "Iris-virginica")
      features = np.array([float(feature) for feature in row[:-1]])
      dataset.append(np.concatenate((features, [label]), axis=0))
  return dataset

