import random
import numpy as np


def get_user_params(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    max_userid = 0
    categories = set()
    for line in lines:
        items = line.strip().split(' ')
        userid = int(items[0])
        max_userid = max(max_userid, userid)
        categories.update(items[1:])

    num_categories = len(categories)

    # data1
    # num_categories = 50
    # data2
    # num_categories = 40
    # num_categories = 9040

    params = np.zeros((max_userid + 1, num_categories))

    for line in lines:
        items = line.strip().split(' ')
        userid = int(items[0])
        user_categories = [int(category) for category in items[1:]]
        user_categories.sort()

        for category in user_categories:
            params[userid, category] = 1
            # params[userid, category] = random.uniform(0, 1)


    return params


def get_item_params(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    max_itemid = 0
    categories = set()
    for line in lines:
        items = line.strip().split(' ')
        itemid = int(items[0])
        max_itemid = max(max_itemid, itemid)
        categories.update(items[1:])

    # num_categories = 50
    # num_categories = 40
    num_categories = len(categories)
    # num_categories = 9040

    params = np.zeros((max_itemid + 1, num_categories))

    for line in lines:
        items = line.strip().split(' ')
        itemid = int(items[0])
        item_categories = [int(category) for category in items[1:]]
        item_categories.sort()

        for category in item_categories:
            params[itemid, category] = 1
            # params[itemid, category] = random.uniform(0, 1)

    return params

# file_path = "item_category.txt"
# params = get_item_params(file_path)
# np.savetxt('item_params_test.txt', params, delimiter=',')
