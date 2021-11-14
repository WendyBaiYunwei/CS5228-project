import random as rd
import collections
import numpy as np


# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items
            """
            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]
            """
    return user_dict_list, item_dict,


class Data:
    # Initialize elements stored in the Data class
    def __init__(self, args):
        # Path to the dataset files
        self.path = './'
        self.train_file = self.path + 'train.txt'
        self.test_file= self.path+"test.txt"
        self.batch_size = args.batch_size

        # Batch size during training
        # self.batch_size = args.batch_size
        # Organized Data used for evaluation
        # self.evalsets = {}

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []


        self.train_user_list = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)
        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0
        
    def load_data(self):

        # Load data into structures
        self.train_user_list, train_item = helper_load(self.train_file)
        self.test_user_list, test_item = helper_load(self.test_file)
        temp_lst = [train_item, test_item]

        self.users = list(set(self.train_user_list.keys()))
        self.items = list(set().union(*temp_lst))
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        for i in range(self.n_users):
            self.n_observations += len(self.train_user_list[i])

    # Sampling batch_size number of users from the training users
    # Each one with one positive observation and one negative observation
    def sample(self):

        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        total = 0

        for user in users:

            if not self.train_user_list[user]:
                pos_items.append(0)
            else:
                index = rd.randint(0, len(self.train_user_list[user])-1)
                pos_items.append(self.train_user_list[user][index])

            while True:

                neg_item = self.items[rd.randint(0, len(self.items)-1)]

                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    break

        return users, pos_items, neg_items