import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import itertools
# def load_data(file_path):
#https://blog.csdn.net/weixin_42641395/article/details/87987802


def read_datas(filename):
    citys = pd.DataFrame(pd.read_excel(filename, header=None))
    citys_cor = np.array(citys.values)
    return citys_cor


#
def compute_dis_mat(location):
    num_city = len(location)
    dis_mat = np.zeros((num_city, num_city))
    for i in range(num_city):
        for j in range(num_city):
            if i == j:
                dis_mat[i][j] = np.inf #
                continue
            a = location[i]
            b = location[j]
            tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
            dis_mat[i][j] = tmp
    return dis_mat

def compute_total_dist(location,order):
    dist_mat = compute_dis_mat(location)
    total_dist = 0
    for i in range(len(order)):  
        if i ==0:
                continue
        p1 = order[i-1]
        p2 = order[i]
        total_dist += dist_mat[p1][p2]
    return total_dist


VERY_LARGE_NUMBER = np.inf
def greedy(start,location):
    '''
    Implementation of the greedy algorithm
    '''
    dist_mat = compute_dis_mat(location)
    n = len(location)
    visited = [0 for e in location] #build a list values 0, with same length of location,记录每个城市被拜访的顺序
    current = start
    visited[current]=1
    for i in range(n):
        min_dist = VERY_LARGE_NUMBER
        min_k = -1
        for k in range(n):
            if visited[k] == 0: #没被拜访的城市
                dist_ = dist_mat[current][k]
                if dist_ < min_dist:
                    min_dist = dist_
                    min_k = k
        current = min_k
        if i+2 <= 100:
            visited[current] = i+2 #在current的基础上，min_k的顺序紧随curent
        else:
            continue

    order = [visited.index(i) for i in range(1,1+n)] #visited: 1~100
    order.append(start)
    return compute_total_dist(location,order),order


def ramdom_start(location):
    n = len(location)
    min_dist = VERY_LARGE_NUMBER
    dist=[]
    min_start = -1
    min_order = []
    for i in range(n):
        start = i
        total_dist_, order = greedy(start,location)
        dist.append(total_dist_)
        if total_dist_< min_dist:
            min_dist = total_dist_
            min_start  = start
            min_order = order

    return min_dist,min_start,min_order,dist


if __name__ == "__main__":

    path = r"Problem I-TSP_100Cities.xlsx"
    location = read_datas(path)  # 2 x num_city
    begin = time.time()

    # length,order = greedy(0,location)
    # start =0
    length,start,order,dist = ramdom_start(location)

    end = time.time()
    print(length)
    print("time is ", end -begin)
    dist = np.array(dist)
    print("mean = ", dist.mean())
    print("median = ", np.median(dist))
    print("worst = ", dist.max())
    # import pdb
    # pdb.set_trace()


    X_locations = [location[i][0] for i in order]
    Y_locations = [location[i][1] for i in order]

    plt.scatter(X_locations, Y_locations,color='b')
    plt.plot(X_locations, Y_locations,"->",color='g')
    plt.scatter(X_locations[0], Y_locations[0],color='r')
    plt.show()
