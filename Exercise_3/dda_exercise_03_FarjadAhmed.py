from array import array
from pkgutil import get_data
from threading import local
from unittest.mock import NonCallableMagicMock
from mpi4py import MPI
import numpy as np
from numpy import genfromtxt
import math
import time
comm = MPI.COMM_WORLD # create a communicator
size = comm.size # get the size of the cluster
rank = comm.rank # get the rank of the process
root = 0 # root process

st = MPI.Wtime() 
np.random.seed(0) 
#Farjad Ahmed_DDA_Exercise_3_KMeans
k = 3 # number of clusters
filename = 'Absenteeism_at_work_AAA/Absenteeism_at_work.csv' # file name

def get_euclidean(a,b): 
    sum_squared = 0 # variable to store the sum of squared differences
    for i in range(len(a)):  # iterate over each element of the array
        sum_squared += math.pow((a[i] - b[i]), 2) # add the squared difference to the sum
    return math.sqrt(sum_squared) # take the square root of the sum to get the euclidean distance

def read_file(file): 
    my_data = genfromtxt(filename, delimiter=';', skip_header = 1) # read the data from the file
    return my_data 

def get_centroids(data, k): # get the centroids
    centroids = np.random.randint(data.shape[0], size=k) # get random centroids
    cents = [] # create a list to store the centroids
    for x in centroids:
        cents.append(np.asanyarray(data[x], dtype = float)) # append the centroids to the list
    return cents # return the list of centroids

def distance_calculation(data_mat, centroids): # calculate the distance between the data and the centroids
    dist_mat = np.zeros((data_mat.shape[0], k)) # create a matrix to store the distances
    for rows in range(data_mat.shape[0]): # iterate over each row of the data matrix
        for p in range(len(centroids)): # iterate over each centroid
            dist_mat[rows][p] = get_euclidean(data_mat[rows], centroids[p]) # calculate the distance between the data and the centroid
    return dist_mat # return the distance matrix

if rank == 0:
    filedata = read_file(filename)  # read the file
    cent_loc = get_centroids(filedata, k) # get the centroids
    sdata = np.array_split(filedata, size, axis=0) # split the data
else:
    sdata = None
    cent_loc = None
    

bdata = comm.bcast(cent_loc, root = root) # broadcast the centroids to all the processes
getdata = comm.scatter(sdata , root = 0) # scatter the data to all the processes
old_global = None # create a variable to store the old centroids
loopflag = True  # create a flag to check if the loop should continue
loop = 0 # create a variable to count the number of iterations


while loopflag: # loop until the flag is false
    
    old_global = bdata # store the old centroids

    distances = distance_calculation(getdata, bdata) # calculate the distance between the data and the centroids
    
    associations = np.array([]) # create a variable to store the associations
    for d in range(len(distances)): # iterate over each row of the distance matrix
        index_min = np.argmin(distances[d]) # get the index of the minimum distance
        associations = np.append(associations, index_min) # append the index to the list of associations 
    

    indices = [] # create a list to store the indices of the data
    for ith in range(k): # iterate over each centroid
        find = np.where(associations==ith) # get the indices of the data that belong to the centroid
        indices.append((find[0])) # append the indices to the list
    
    local_centroids = [] # create a list to store the local centroids
    for ks in range(k): # iterate over each centroid
        local_centroids.append(np.zeros(getdata[0].shape)) # append a zero array to the list
    local_centroids = np.array(local_centroids) # convert the list to an array

    local_centroids_lengths = np.zeros(len(indices)) # create a list to store the lengths of the local centroids
    for a in range(len(indices)): # iterate over each centroid
        temp1 = [] # create a list to store the data that belongs to the centroid
        for i in indices[a]: # iterate over each index
            temp1.append(np.asarray(getdata[i]))  # append the data to the list
        local_centroids[a] = np.sum(temp1, axis = 0) # calculate the sum of the data and append it to the local centroid
        local_centroids_lengths[a] = len(temp1) # calculate the length of the data and append it to the local centroid

    reduced_points = comm.reduce(local_centroids, root = root) # reduce the local centroids to the root process
    reduced_lenghts = comm.reduce(local_centroids_lengths, root = root) # reduce the local centroids lengths to the root process
    
    
    if reduced_points is not None:  # if the process is not the root process
        if reduced_lenghts is not None: # if the process is not the root process
            global_centroids = [] # create a list to store the global centroids
            for x in range(len(reduced_points)): # iterate over each centroid
                avg = reduced_points[x]/reduced_lenghts[x] # calculate the average of the data and append it to the global centroid
                global_centroids.append(avg) # append the average to the list
    else:
        global_centroids = None # if the process is the root process, set the global centroids to None

    bdata = comm.bcast(global_centroids, root = root) # broadcast the global centroids to all the processes
    a_bdata = np.concatenate(bdata) # concatenate the global centroids
    b_old_global = np.concatenate(old_global) # concatenate the old global centroids
    
    if np.all(np.equal(a_bdata, b_old_global)): # if the global centroids are equal to the old global centroids
        loopflag = False # set the flag to false
        getasso = comm.gather(associations, root = root) # gather the associations to the root process
        if getasso is not None: # if the process is not the root process
            getasso = np.concatenate(getasso) # concatenate the associations
            print('all associations are \n', getasso) # print the associations
    else:
        loop += 1 # if the global centroids are not equal to the old global centroids, increment the loop counter

if rank == root: # if the process is the root process
    et = MPI.Wtime() # get the end time
    serialTime = 0.24791485400000002  # set the serial time
    diff = et-st # calculate the difference between the start and end time
    print('time taken is', diff) # print the difference
    Sp = serialTime/diff # calculate the speedup
    print('Sp is: ', Sp) # print the speedup
    print('loops count: ', loop) # print the number of iterations




