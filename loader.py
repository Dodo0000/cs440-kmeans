import csv

def load_dataset():
  dataset = []
  with open('iris-dataset.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      dataset.append( (float(row[0]), float(row[1]), float(row[2])) )
  return dataset

