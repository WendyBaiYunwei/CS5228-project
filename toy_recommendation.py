import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import csv
import re


df = pd.read_csv('train.csv')


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



def get_recommendation(query):
    print(cv.transform([query]))
    query_feature=tfidf_transformer.transform(cv.transform([query]))
    cos=cosine_similarity(query_feature,tfidfs)[0]
    print(cos)
    df['similarity']=cos

    # df['similarity']=df.apply(get_similarity,axis=1)
    result=df.sort_values(by=['similarity'],ascending=False)
    return result

res=get_recommendation(query)

print(res.head(10))