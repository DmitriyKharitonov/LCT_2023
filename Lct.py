import pandas as pd
import seaborn as sns
import numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from google.colab import drive
from sklearn.ensemble import GradientBoostingClassifier


drive.mount('/content/drive/')
test_data = pd.read_parquet('/content/drive/MyDrive/hackathon_files_for_participants_ozon/test_data.parquet')
test_pairs = pd.read_parquet('/content/drive/MyDrive/hackathon_files_for_participants_ozon/test_pairs_wo_target.parquet')
full_data = pd.read_parquet('/content/drive/MyDrive/hackathon_files_for_participants_ozon/train_data.parquet')
train_pairs = pd.read_parquet('/content/drive/MyDrive/hackathon_files_for_participants_ozon/train_pairs.parquet') 

full_data_name = full_data[['variantid', 'name_bert_64']]

train_pairs_name = pd.merge(train_pairs, full_data_name, left_on = 'variantid1', right_on = 'variantid', how = 'inner', suffixes=("", ""))
train_pairs_name = train_pairs_name.drop(columns = ['variantid']).rename(columns = {'name_bert_64' : 'name_1'})
train_pairs_name = pd.merge(train_pairs_name, full_data_name, left_on = 'variantid2', right_on = 'variantid', how = 'inner', suffixes=("", ""))
train_pairs_name = train_pairs_name.drop(columns = ['variantid']).rename(columns = {'name_bert_64' : 'name_2'})

train_pairs_name['full'] = train_pairs_name['name_1'] - train_pairs_name['name_2']

X_name = train_pairs_name['full'].to_list()
y_name = train_pairs_name['target']

clf_name = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=0, verbose = True).fit(X_name, y_name)
clf_name.score(X_name, y_name)

full_test_name = test_data[['variantid', 'name_bert_64']]

test_pairs_name = pd.merge(test_pairs, full_test_name, left_on = 'variantid1', right_on = 'variantid', how = 'inner', suffixes=("", ""))
test_pairs_name = test_pairs_name.drop(columns = ['variantid']).rename(columns = {'name_bert_64' : 'name_1'})
test_pairs_name = pd.merge(test_pairs_name, full_test_name, left_on = 'variantid2', right_on = 'variantid', how = 'inner', suffixes=("", ""))
test_pairs_name = test_pairs_name.drop(columns = ['variantid']).rename(columns = {'name_bert_64' : 'name_2'})

test_pairs_name['full'] = test_pairs_name['name_1'] - test_pairs_name['name_2']
#test_pairs_name['full'] = test_pairs_name['full'].to_list()

X_test_name = test_pairs_name['full'].to_list()
pred2 = clf_name.predict_proba(X_test_name)

pred_class_2 = []

for i in range(len(pred2)):
  pred_class_2.append(pred2[i][1])


test_pairs_name['target'] = pred_class_2
answer2 = test_pairs_name[['variantid1', 'variantid2', 'target']]
answer2.to_csv('answ_name.csv')