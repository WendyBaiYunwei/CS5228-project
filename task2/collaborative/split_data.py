import csv
import json

usr_cnt=0
user_dict={}
item_dict={}
interaction=0
with open('fabricated_interactions.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for cnt,row in enumerate(reader):
        if cnt==0:
            continue
        if len(row)==1:
            continue
        else:
            user=usr_cnt
            items=eval(row[1])
            user_dict[user]=items
            interaction+=len(items)
            for item in items:
                if item not in item_dict:
                    item_dict[item]=[]
                item_dict[item].append(user)
            usr_cnt+=1

print("number of users:",len(user_dict))
print("number of items:",len(item_dict))
print("number of interaction:",interaction)
print("sparsity:",interaction/len(user_dict)/len(item_dict))

test_weight=1
train_weight=8

train_dict={}
test_dict={}
train_size=0
test_size=0

for user,items in user_dict.items():
    train_items=[]
    test_items=[]
    if len(items)<test_weight+train_weight:
        test_dict[user]=items[:1]
        train_dict[user]=items[1:]

        test_size+=len(test_dict[user])
        train_size+=len(train_dict[user])

    
    else:
        split=int(len(items)*test_weight/(test_weight+train_weight))+1
        test_dict[user]=items[:split]
        train_dict[user]=items[split:]
        test_size+=len(test_dict[user])
        train_size+=len(train_dict[user])

# print(train_size)
# print(test_size)
# with open("train.json","w") as f:
#     json.dump(train_dict,f)

# with open("test.json","w") as f:
#     json.dump(test_dict,f)


with open("test.txt",'w') as f:
    for key,value in test_dict.items():
        st=str(key)
        for item in value:
            st+=" "+str(item).strip("\n")
        f.write(st+"\n")

with open("train.txt",'w') as f:
    for key,value in train_dict.items():
        st=str(key)
        for item in value:
            st+=" "+str(item).strip("\n")
        f.write(st+"\n")
