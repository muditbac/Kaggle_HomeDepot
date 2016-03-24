import pandas as pd
from local_paths import *

# Native implementation using csv library
# import csv
# def load_csv(filename):
# 	data = []
# 	with open(INPUT_PATH + filename,'r') as f:
# 		reader = csv.reader(f)
# 		for row in reader:
# 			data.append(row)
# 	return data


train = pd.read_csv(INPUT_PATH + "train.csv", encoding="ISO-8859-1")[:1000]
test = pd.read_csv(INPUT_PATH + "test.csv", encoding="ISO-8859-1")[:1000]
attributes = pd.read_csv(INPUT_PATH + "attributes.csv", encoding="ISO-8859-1")[:50000]
descriptions = pd.read_csv(INPUT_PATH + "product_descriptions.csv", encoding="ISO-8859-1")[:1000]

