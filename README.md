# AOA_experiments
Code for AOA course
# AOA_experiments
Experiments of my AOA class

Data: 
  Problem I-TSP_100Cities.xlsx locations of 100 cities
  
  Problem II-Virtual_TSP100Cities.xlsx cost of edges between cities
  
Code:
  1. AOA_TSP_greedy: 
     greedy for problem I. load locations ————> calculate the distance matrix ————> greedy ————> plot
     
  2. AOA_TSP_greedy_VP: 
     greedy for problem II.load costs ————> trans to distance matrix ————> follow 1
     
  3. TSP_localsearch: two-edge change (inversion) neibors, first move, early stop.
     import greedy from 1 and 2 to abtain initial values————> select initial solutions (best 20%) ————> two edge change neigbor and evaluate ————> iterate until no better solution is obtained (for 3 iterations).
