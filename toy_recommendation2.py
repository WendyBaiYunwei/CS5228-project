import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import csv
import re
from toy_recommendation import get_similar
import random

TTL_USERS = 3000
RATIO = 3000 * 634042
TTL_MAKE = np.array([])
TTL_AGE = np.array([])
TTL_CC = np.array([9424 + 361261, 180982 + 69630, 12745]) * RATIO #0-1600, 1601-3000, 3000+
type = 1

def init():
    # assume each person just have one car in mind
    d = {}
    step = (max_price - min_price) // TTL_USERS ## visualize and obtain suitable max and min
    uid = 0
    needed_make = {}#make list
    needed_age = {}
    needed_cc = {}
    for i, make in enumerate(make_list): ## read xml
        needed_make[make] = TTL_MAKE[i]
    for age in range(21):
        needed_age[age] = TTL_AGE[age]
    for age in range(3):
        needed_cc[age] = TTL_CC[age]

    for price in range(min_price, max_price, step):
        i = 0
        while i <= 100000:
            item_by_price ##
            item_id_by_price = item_by_price.index ## xxx., find matching item by price then remove the item
            make_type = item_by_price.make
            age_type = item_by_price.age
            if age_type > 20:
                age_type = 21
            cc_type = item_by_price.engine_cap
            if cc_type <= 1600:
                cc_type = 0
            elif cc_type <= 3000:
                cc_type = 1
            else:
                cc_type = 2
            if needed_make[make_type] > 0 and needed_age[age_type] > 0 and needed_cc[cc_type] > 0:
                needed_make[make_type] -= 1
                needed_age[age_type] -= 1
                needed_cc[cc_type] -= 1
                break
            i += 1
        d[uid] = item_id_by_price
        uid += 1
    return d[uid]

def get_items(uid, base_items, base_item_i):
    ttl_user_interacts = u_to_intersize[uid]
    similar_for_each_base = ttl_user_interacts // len(base_items)
    res = get_similar(base_items, base_item_i, size=similar_for_each_base)
    return res.index.tolist().extend(base_item_i)

def writeCSV(users):
    with open('fabricated_interaction.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['user_id', 'Interactions'])

        for user_id in users:
            u_to_bs = init()
            base_items_i, base_items = u_to_bs[user_id]
            browsed_items = get_items(user_id, base_items_i, base_items)
            w.writerow([str(user_id), str(browsed_items)])

def fill_intersize(users):
    u_to_intersize = {}
    # normal distribution with max = 50, min = 0
    mean, sd = 10, 0.1 # mean and standard deviation
    inters = np.random.normal(mean, sd, 3000)
    random.shuffle(inters)
    for i, uid in enumerate(users):
        u_to_intersize[uid] = inters[i]

if __name__ == '__main__':
    users = [i for i in range(TTL_USERS)]
    u_to_intersize = fill_intersize(users)
    writeCSV(users, u_to_intersize)