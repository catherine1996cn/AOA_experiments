from AOA_TSP_greedy import read_datas, compute_dis_mat, ramdom_start,compute_total_dist
from AOA_TSP_greedy_VP import read_datas2, compute_dis_mat2, ramdom_start2
import matplotlib.pyplot as plt
import numpy as np
import  random
import time

#initial
def initial1():
    path = r"Problem I-TSP_100Cities.xlsx"
    location = read_datas(path)  # 2 x num_city
    dist_mat = compute_dis_mat(location)
    length,start,order,_ = ramdom_start(location)
    return location, dist_mat,length,start,order

def initial2():
    filename = r"Problem II-Virtual_TSP100Cities.xlsx"
    location = read_datas2(r"Problem I-TSP_100Cities.xlsx")
    dist_mat = compute_dis_mat2(read_datas2(filename)) #different from 1
    length, start, order, _ = ramdom_start2(filename,location)
    return location, dist_mat, length, start, order


def TSP_plt(location,order):
    X_locations = [location[i][0] for i in order]
    Y_locations = [location[i][1] for i in order]

    plt.scatter(X_locations, Y_locations, color='b')
    plt.plot(X_locations, Y_locations, "->", color='g')
    plt.scatter(X_locations[0], Y_locations[0], color='r')
    plt.show()
    return



def get_edge_and_evaluate(order,dist_mat,length):
    get_new = False
    evaluate_iter = 0
    new_order = []
    while  get_new == False:
        end1,end2 = random.sample(order,2)
        n_inverse = abs(end1-end2)
        if n_inverse ==1:
            continue

        evaluate_iter +=1
        if evaluate_iter > 4000:
            break


        x1=min(end1,end2)
        x2=max(end1,end2)

        edge1 = [x1-1,x1]
        edge2 = [x2,x2+1]
        if x1 == 0:
            edge1[0] = len(order)-1
        if x2 == len(order)-1:
            edge2[1]=0

        #edge1[0]-edge1[1]-edge2[0]-edge2[1] index in order[]
        #x10--x11--x20--x21
        x10,x11 = order[edge1[0]],order[edge1[1]]
        x20,x21 = order[edge2[0]],order[edge2[1]]
        length_minor = dist_mat[x10][x11] + dist_mat[x20][x21]
        length_add = dist_mat[x10][x20] + dist_mat[x11][x21]

        length_change = length_add - length_minor


        if length_change < 0:
        # revise i2--j1

            if edge1[1] == 0:
                list1=[]
            else:
                list1 = order[:edge1[1]]

            reverse_list = order[edge1[1]:edge2[0] + 1]
            reverse_list.reverse()

            if edge2[0] == 0:
                list3 =[]
            else:
                list3 = order[edge2[0]+1:]

            new_order = list1 + reverse_list + list3
            new_length = length+length_change
            get_new = True


        else:
            length_change=0
            new_order = order
            new_length = length

    return length_change, new_order, new_length, evaluate_iter




def pltshow(location,order):
    order = order + [order[0]]
    X_locations = [location[i][0] for i in order]
    Y_locations = [location[i][1] for i in order]

    plt.scatter(X_locations, Y_locations, color='b')
    plt.plot(X_locations, Y_locations, "->", color='g')
    plt.scatter(X_locations[0], Y_locations[0], color='r')
    plt.show()



def local_search1(order,location, length,dist_mat,start_time):
    evaluate_total = 0
    for_stop = 0
    while evaluate_total < 100000:
        evaluate_total +=1

        evaluate_result = get_edge_and_evaluate(order,dist_mat,length)

        if evaluate_result is not None:
            length_change,new_order, new_length, evaluate_iter = evaluate_result

            evaluate_total += evaluate_iter
            print("length_change", new_length-length)
            print("evaluate_time", evaluate_total)
            print("time is ", time.time() - start_time)
            print("current_best_length", new_length)
            print("best_order", new_order)

            if new_length - length ==0:
                for_stop += 1
                if for_stop >3:
                    break
            # order_to_calcu = new_order+[new_order[0]]
            # calculate_length = compute_total_dist(location, order_to_calcu)
            # print('calculate_length:', calculate_length)

            pltshow(location, order)

        else:
            new_order = order
            new_length = length

        order = new_order
        length = new_length

    pltshow(location, order)

    print("No better tour is obtained")
    # print("best_order", order)
    # print("best_length", length)

    return order, length





if __name__ == "__main__":

    # location, dist_mat,length,start,order = initial1()
    location, dist_mat,length,start,order = initial2()

    print(order)
    print(length)
    order.pop()
    start_time = time.time()
    order,length=local_search1(order, location,length,dist_mat,start_time)

