import csv
import numpy as np

def load_dataset():
  dataset = []
  with open('iris-dataset.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      if len(row) == 0:
        continue
      dataset.append( np.array([float(feature) for feature in row[:-1]]) )
  return dataset

