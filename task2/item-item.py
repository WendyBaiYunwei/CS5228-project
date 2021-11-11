import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import csv
import re

R_SIZE = 5 # number of items to recommend

df = pd.read_csv('../task1/ml_approach/data/train_basic.csv')
df['title']=df['title'].fillna('')
df['description']=df['description'].fillna('')
df['features']=df['features'].fillna('')
df['accessories']=df['accessories'].fillna('')

def create_soup(x):
    return ''.join(x['title'])+' '+''.join(x['description'])+' '+''.join(x['features'])+' '+''.join(x['accessories'])

# preprocess words
df['soup'] = df.apply(create_soup,axis=1)
cv = CountVectorizer(stop_words='english')
word_cnt = cv.fit_transform(df['soup'])
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_cnt)
tfidfs=tfidf_transformer.transform(cv.transform(df['soup']))
query="A big comfortable car with 1 owner and 320i gt m-sports model"

def get_similar(base_result_i, size):
    useful_features = ['price', 'age']
    base_result_features = np.array(df.loc[base_result_i, useful_features]).reshape(1, -1)
    pair_wise_sim = cosine_similarity(df[useful_features], base_result_features)
    df['similarity_num'] = pair_wise_sim
    res = df.sort_values(by = ['similarity_num'],ascending=False)
    res = res[res.type_of_vehicle == df.loc[base_result_i, 'type_of_vehicle']]
    res = res[['title', 'price', 'age']]
    return res[:size]

def get_base_item(query):
    query_feature = tfidf_transformer.transform(cv.transform([query]))
    cos = cosine_similarity(query_feature,tfidfs)[0]
    df['similarity'] = cos

    base_result_i = df.sort_values(by=['similarity'],ascending=False).index.tolist()[0]
    return base_result_i

base_result_i = get_base_item(query)
extended = get_similar(base_result_i, R_SIZE)

print(extended)