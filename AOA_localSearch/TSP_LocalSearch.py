from AOA_TSP_greedy import read_datas, compute_dis_mat, ramdom_start,compute_total_dist
from AOA_TSP_greedy_VP import read_datas2, compute_dis_mat2, ramdom_start2
from AOA_TSP_greedy import greedy as greedy1
from AOA_TSP_greedy_VP import greedy2
import matplotlib.pyplot as plt
import numpy as np
import  random
import time
import openpyxl
import pandas as pd
import os

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
    length_change = 0
    new_order = order
    new_length = length
    while  get_new == False:
        end1,end2 = random.sample(order,2)
        n_inverse = abs(end1-end2)
        if n_inverse ==1:
            continue

        evaluate_iter +=1
        if evaluate_iter > 6000:
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


        # else:
        #     length_change=0
        #     new_order = order
        #     new_length = length

    return length_change, new_order, new_length, evaluate_iter




def pltshow(location,order,version):
    order = order + [order[0]]
    X_locations = [location[i][0] for i in order]
    Y_locations = [location[i][1] for i in order]

    plt.scatter(X_locations, Y_locations, color='b')
    plt.plot(X_locations, Y_locations, "->", color='g')
    plt.scatter(X_locations[0], Y_locations[0], color='r')
    plt.title(version)
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
            # print("length_change", new_length-length)
            # print("evaluate_time", evaluate_total)
            # print("time is ", time.time() - start_time)
            print("cureent_compute", evaluate_total)
            print("current_best_length", new_length)
            print("best_order", new_order)

            if new_length - length ==0:
                for_stop += 1
                if for_stop >3:
                    break

            # pltshow(location, order)

        else:
            new_order = order
            new_length = length

        order = new_order
        length = new_length

    compute_time = time.time()-start_time
    # pltshow(location, order，version)

    print("No better tour is obtained by local search")

    return order, length, evaluate_total,compute_time

def list2str(input):
    """
    input: 1xm list or np.array
    output: str item0——>item1——> ...——>item -1
    """
    input = list(input)
    input = [str(i) for i in input]
    output = " ".join(input)
    return output

def str2list(input):

    # input = str.replace(input,"——>",",")
    input = input.split(" ")
    output = [int(i) for i in input]
    # output = np.array(input)
    return output


def save_csv(save_name,version:str,data,names=None,mode="w",header=False):
    # import pdb
    # pdb.set_trace()

    if names is None:
        names=["version","length","tour","single_compute","single_time","total_compute","total_time"]
    else:
        names = names
    csv_data = []
    data.insert(0,version)
    csv_data.append(data)
    csv_pd = pd.DataFrame(csv_data,columns=names)
    csv_pd.to_csv(save_name,sep=',',mode=mode, header=header,index=False)

def LocalSearch_MultiIntial(filename,save_name,n=100,problem=1):
    evaluate_total = 0
    compute_time = 0
    for i in range(n):
        start = i

        if problem == 1:
            location = read_datas(filename)
            dist_mat = compute_dis_mat(location)
            init_length,init_order, = greedy1(start,location)
        else:
            location = read_datas2(filename)
            dist_mat = compute_dis_mat2(location)
            init_length,init_order = greedy2(start,filename)

        init_order.pop()
        start_time = time.time()
        order, length, evaluate_single,single_time = local_search1(init_order, location, init_length, dist_mat, start_time)
        version = "Problem{}".format(problem)+ "_start_from_city{}".format(start)
        tour = list2str(order)
        evaluate_total += evaluate_single
        compute_time += single_time
        data = [length,tour,evaluate_single,single_time,evaluate_total,compute_time]
        save_csv(save_name,version, data, mode='a')

        file_location = r"Problem I-TSP_100Cities.xlsx"
        plt_location =read_datas(file_location)
        pltshow(plt_location, order,version+"with_length_"+str(length))

def greedy_initial(filename,save_name,n=100,problem=1):
    for i in range(n):
        start = i

        if problem == 1:
            location = read_datas(filename)
            # dist_mat = compute_dis_mat(location)
            init_length, init_order, = greedy1(start, location)
        else:
            location = read_datas2(filename)
            # dist_mat = compute_dis_mat2(location)
            init_length, init_order = greedy2(start, filename)

        init_order.pop()
        version = "Problem{}".format(problem) + "_start_from_city{}".format(start)
        tour = list2str(init_order)
        data = [init_length, tour]
        names=["version","length",'tour']
        save_csv(save_name, version, data=data, names=names , mode='a')

def save_initials_from_greedy(problem=1):
    filename1 = r"Problem I-TSP_100Cities.xlsx"
    filename2 = r"Problem II-Virtual_TSP100Cities.xlsx"

    save_name1 = "greedy_initial_1.csv"
    save_name2 = "greedy_initial_2.csv"
    greedy_initial(filename1, save_name1, n=100, problem=1)
    greedy_initial(filename2, save_name2, n=100, problem=2)



def experiment_1():
    """
    strat from all cities to obtain 100 initial tours,
    then run local search for each initial solution
    """
    filename1 = r"Problem I-TSP_100Cities.xlsx"
    filename2 = r"Problem II-Virtual_TSP100Cities.xlsx"

    # LocalSearch_MultiIntial(filename1, n=100, problem=1)
    save_name = "good_results_problem1_try3.csv"
    LocalSearch_MultiIntial(filename1, save_name, n=100, problem=1)

def experiment_2(i=1,problem=1):
    """
    strat from all cities and select 20% best solutions as initial solutions.
    then run local search for each initial solution
    """

    filename1 = r"Problem I-TSP_100Cities.xlsx"
    filename2 = r"Problem II-Virtual_TSP100Cities.xlsx"
    initial_file1 = "greedy1_initial_20.csv"
    initial_file2 = "greedy2_initial_20.csv"
    location =read_datas(filename1)


    if problem == 1:
        initial_file = initial_file1
        dist_mat = compute_dis_mat(read_datas(filename1))
    else:
        initial_file = initial_file2
        dist_mat = compute_dis_mat2(read_datas2(filename2))

    initial_solutions = pd.DataFrame(pd.read_csv(initial_file, header=None))
    evaluate_total = 0
    compute_time = 0
    lengths = dict()
    lengths["samll_lengthes"] = []
    lengths["save_names"] = []
    smallest_length = np.inf
    small_name=None
    for data in initial_solutions.values:
        version, init_length, tour = data
        init_order = str2list(tour)
        init_length = float(init_length)

        start_time = time.time()
        order, length, evaluate_single,single_time = local_search1(init_order, location, init_length, dist_mat, start_time)
        evaluate_total += evaluate_single
        compute_time += single_time

        order.pop()
        tour = list2str(order)
        data = [length, tour, evaluate_single,single_time,evaluate_total,compute_time]
        names=["version","length","tour","single_compute","single_time","total_compute","total_time"]

        save_name = "problem{}".format(problem)+"_20initial_try{}".format(i)+".csv"
        save_name = os.path.join("result3",save_name)
        save_csv(save_name=save_name,version=version, data=data, names=names, mode='a')

        if length<smallest_length:
            smallest_length = length
            small_name = save_name

    lengths["samll_lengthes"].append(smallest_length)
    lengths["save_names"].append(small_name)

    return lengths



if __name__ == "__main__":

    # for i in range(1000):
    #     lengths = experiment_2(i, problem=2)
    #     small_lengthes = lengths["samll_lengthes"]
    #     small_lengthes = list2str(small_lengthes)
    #     save_names = lengths["save_names"]
    #     save_names = list2str(save_names)
    #     if i == 0:
    #         mode = "w"
    #     else:
    #         mode = "a"
    #     save_name = os.path.join("result3", "Ro_problem_{}.csv".format(2))
    #     save_csv(save_name=save_name, version="6000", data=[small_lengthes, save_names],
    #              names=["earlystop", "samll_lengthes", "save_names"], mode=mode, header=False)
    #
    # for i in range(1000):
    #     lengths=experiment_2(i,problem=1)
    #     small_lengthes = lengths["samll_lengthes"]
    #     small_lengthes = list2str(small_lengthes)
    #     save_names = lengths["save_names"]
    #     save_names = list2str(save_names)
    #     if i==0:
    #         mode = "w"
    #     else:
    #         mode ="a"
    #     save_name = os.path.join("result3","Ro_problem_{}.csv".format(1))
    #     save_csv(save_name=save_name, version="6000", data=[small_lengthes,save_names], names = ["earlystop","samll_lengthes","save_names"], mode = mode, header = False)

    input1 = "11 49 59 35 54 91 48 89 52 30 74 28 66 67 68 86 46" \
             " 18 96 53 71 83 10 0 64 55 12 75 50 88 72 69 21 97 " \
             "25 32 70 41 29 6 84 16 99 39 47 27 63 15 8 19 22 62 " \
             "33 24 65 3 23 31 26 92 40 42 58 20 4 85 56 60 36 81 57 " \
             "1 61 37 43 13 93 51 87 78 98 9 44 34 90 14 94 2 80 7 " \
             "95 77 38 5 45 17 73 79 76"


    input2 = "50 18 4 74 13 16 61 36 80 44 62 29 82 76 49 " \
             "71 94 43 47 59 3 38 66 89 41 65 54 78 19 81 90 " \
             "45 21 17 56 0 22 6 32 72 63 84 8 25 75 42 86 83 " \
             "40 51 92 57 60 39 24 7 64 97 28 37 96 68 99 26 20 " \
             "91 2 53 14 34 58 11 12 15 88 1 70 10 67 27 69 48 " \
             "95 9 23 30 5 31 79 33 46 98 87 73 35 85 52 55 77"



    filename1 = r"Problem I-TSP_100Cities.xlsx"
    location = read_datas(filename1)
    order1= str2list(input1)
    order2= str2list(input2)
    pltshow(location=location,order=order1,version="Problem_I_length=79.880")
    pltshow(location=location,order=order2,version="Problem_II_length=83.033")

    # location, dist_mat,length,start,order = initial1()
    # location, dist_mat,length,start,order = initial2()
    # greedy_order1 =[34, 44, 9, 98, 78, 87, 51, 93, 13, 37, 43, 61, 57, 81, 36, 60, 56, 85, 4, 20, 55, 64, 0, 10, 83, 71, 96, 18, 68,
    #  86, 46, 67, 66, 28, 74, 30, 52, 89, 48, 91, 54, 35, 59, 49, 82, 11, 53, 12, 75, 50, 88, 72, 69, 21, 25, 97, 58, 42,
    #  29, 6, 84, 70, 41, 32, 16, 99, 39, 47, 27, 63, 23, 3, 65, 24, 33, 62, 22, 19, 8, 15, 26, 92, 40, 31, 1, 76, 79, 73,
    #  17, 45, 5, 38, 77, 94, 2, 95, 7, 80, 14, 90, 34]
    # greedy_length1 = 88.19310392078192
    #
    # greedy_order2 = [51, 92, 57, 60, 7, 24, 70, 10, 49, 71, 94, 43, 47, 29, 62, 44, 80, 64, 55, 52, 85, 35, 23, 30, 5, 31, 79, 33, 46,
    #  98, 87, 96, 37, 86, 42, 75, 25, 8, 84, 89, 41, 36, 20, 83, 40, 48, 69, 27, 67, 34, 14, 53, 38, 3, 15, 12, 11, 58,
    #  56, 0, 22, 6, 32, 72, 63, 90, 45, 21, 17, 50, 18, 4, 74, 13, 16, 61, 2, 9, 95, 65, 54, 78, 19, 81, 1, 88, 26, 99,
    #  68, 73, 77, 93, 97, 28, 91, 66, 76, 82, 59, 39, 51]
    # greedy_length2 = 94.79069991123225
    # input = "53 96 18 68 86 46 72 88 50 75 12 71 83 10 0 64 55 85 4 " \
    #       "20 42 58 97 25 21 69 32 41 70 6 84 29 16 99 39 47 27 63 15 " \
    #       "8 19 22 62 33 24 65 3 23 40 92 26 31 1 56 60 36 81 57 61 37 " \
    #       "43 13 93 51 87 78 98 9 34 44 14 90 38 77 94 2 80 7 95 5 45 76 " \
    #       "79 73 17 89 48 91 11 82 49 59 35 54 52 30 74 28 66 53"
    # input="0 64 55 20 4 85 56 60 36 81 57 1 31 26 92 40 42 58 97 25 21 " \
    #       "69 32 41 70 6 84 29 16 99 39 47 27 63 23 3 65 24 33 62 22 19 8 " \
    #       "15 87 51 93 13 9 98 44 34 90 14 94 77 38 2 95 7 80 5 45 76 79 " \
    #       "73 17 89 48 91 54 35 59 49 82 11 10 83 71 96 18 68 86 46 67 66" \
    #       " 28 74 30 52 53 12 75 50 88 72 61 37 43 78 0"
    # order = str2list(input)

    # filename1 = r"Problem I-TSP_100Cities.xlsx"
    # location = read_datas(filename1)
    # dist_mat = compute_dis_mat(location)
    # total_dist = 0
    # for i in range(len(order)):
    #     if i == 0:
    #         continue
    #     p1 = order[i - 1]
    #     p2 = order[i]
    #     total_dist += dist_mat[p1][p2]
    # print(total_dist)

