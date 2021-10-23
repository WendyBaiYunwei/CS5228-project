import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import csv
import re

R_SIZE = 5 #items to recommend through each method

df = pd.read_csv('numerical_cleaned.csv')


useless=['listing_id','indicative_price','opc_schem']
natural_language=['title','description','features','accessories']
single_catagorical={'make':78,'transmission':2,'fuel_type':5,'no_of_owners':7,'type_of_vehicle':0}
multi_catagorical=['category']
numerical=['manufactured','original_reg_date','reg_date','curb_weight','power','engine_cap','mileage','lifespan']
numerical_price=['coe','road_tax','dereg_value','omv','arf']
output=['price']

print(df.shape)


df['title']=df['title'].fillna('')

df['description']=df['description'].fillna('')

df['features']=df['features'].fillna('')

df['accessories']=df['accessories'].fillna('')

def create_soup(x):
    return ''.join(x['title'])+' '+''.join(x['description'])+' '+''.join(x['features'])+' '+''.join(x['accessories'])


df['soup']=df.apply(create_soup,axis=1)

#tfidf=TfidfVectorizer(stop_words='english')
cv=CountVectorizer(stop_words='english')
word_cnt=cv.fit_transform(df['soup'])
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_cnt)
tfidfs=tfidf_transformer.transform(cv.transform(df['soup']))
query="A big comfortable car with 1 owner and 320i gt m-sports model"

def get_similar(base_result, base_result_i):
    useful_features = df[['price', 'age', 'mileage']]
    base_result_features = useful_features.iloc[base_result_i.index.to_list()]
    pair_wise_sim = cosine_similarity(useful_features, base_result_features)
    w_price = 5 # based on assumptions
    w_age = 2
    w_mileage = 2 
    df['similarity_num']=pair_wise_sim[:, 0] * w_price  +\
        pair_wise_sim[:, 1] * w_age + \
            pair_wise_sim[:, 4] * w_mileage
    res=df.sort_values(by=['similarity_num', 'similarity'],ascending=False)
    valid_types = list(base_result['type_of_vehicle'].values)
    is_valid = res.type_of_vehicle.isin(valid_types)
    res=res[is_valid]
    return res[:R_SIZE]

def get_base_recommendation(query):
    query_feature=tfidf_transformer.transform(cv.transform([query]))
    cos=cosine_similarity(query_feature,tfidfs)[0]
    df['similarity']=cos

    base_result=df.sort_values(by=['similarity'],ascending=False)
    base_result_i=df.sort_values(by=['similarity'],ascending=False)
    return (base_result[:R_SIZE], base_result_i[:R_SIZE])

base_result, base_result_i=get_base_recommendation(query)
extended = get_similar(base_result, base_result_i)

res = [base_result, extended]
res = pd.concat(res)
print(res)