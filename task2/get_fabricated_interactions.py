import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import csv
import re
import random

TTL_USERS = 3000

def get_similar(base_result, base_result_i, size):
    useful_features = df[['price', 'age', 'mileage']]
    base_result_features = np.array(useful_features.iloc[base_result_i]).reshape(1, -1)
    pair_wise_sim = cosine_similarity(useful_features, base_result_features)
    df['similarity_num']=pair_wise_sim
    res=df.sort_values(by=['similarity_num'],ascending=False)
    res=res[res.type_of_vehicle == base_result['type_of_vehicle']]
    return res[:size]

def init():
    # assume each person just have one car in mind
    d = {}
    uid = 0
    ttl_cars = len(df)
    b_items = []
    step = 20000
    min_price = df['price'].min()
    max_price = df['price'].max()
    for price_l in range(int(min_price), int(max_price) - step, step):
        price_h = price_l + step
        cur_cars = df[df.price >= price_l]
        cur_cars = cur_cars[cur_cars.price <= price_h].index.tolist()
        ttl_cur_cars = len(cur_cars)
        ttl_user = ttl_cur_cars * TTL_USERS // ttl_cars
        # randomly select ttl_user items
        if ttl_cur_cars <= 1:
            continue
        items = random.sample(cur_cars, ttl_user + 1)
        b_items.extend(items)

    for uid in users:
        d[uid] = b_items[uid]
    return d

def get_items(uid, base_items, base_item_i, u_to_intersize):
    ttl_user_interacts = u_to_intersize[uid]
    res = get_similar(base_items, base_item_i, size=ttl_user_interacts)
    res = res.index.tolist()
    return res

def writeCSV(users, u_to_intersize):
    with open('fabricated_interactions.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['user_id', 'Interactions'])
        u_to_bs = init()
        for user_id in users:
            print(user_id)
            base_item_i = u_to_bs[user_id] ##base item
            base_item = df.iloc[base_item_i]
            browsed_items = get_items(user_id, base_item, base_item_i, u_to_intersize)
            w.writerow([str(user_id), str(browsed_items)])

def fill_intersize(users):
    u_to_intersize = {}
    # normal distribution with max = 50, min = 0
    mean, sd = 0, 1 # mean and standard deviation
    inters = np.random.normal(mean, sd, 3000)
    def normal(x,mu,sigma):
        return (2.*np.pi*sigma**2.)**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2.) * 100
    y = normal(inters,mean,sd) 
    random.shuffle(y)
    for i, uid in enumerate(users):
        u_to_intersize[uid] = int(y[i])
    return u_to_intersize

if __name__ == '__main__':
    df = pd.read_csv('../task1/ml_approach/data/train_basic.csv')
    users = [i for i in range(TTL_USERS)]
    u_to_intersize = fill_intersize(users)
    writeCSV(users, u_to_intersize)