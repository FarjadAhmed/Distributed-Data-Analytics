from collections import defaultdict
from operator import invert
from turtle import color
from typing import Counter
from mpi4py import MPI
import numpy as np
import cv2
from itertools import groupby
import matplotlib.pyplot as plt
st = MPI.Wtime()
bucket = np.zeros(256)
imageName = 'bigpic.png'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

def mycounter(lst): # This is a function to count the number of times each intensity appears in the image
    mydict = {} # This is a dictionary to store the number of times each intensity appears
    for i1 in lst: # This is a loop to iterate through the intensities
        mydict[i1] = 0 # This is to initialize the dictionary with 0
        for i2 in lst: # This is a loop to iterate through the intensities
            if i1 == i2: # This is to check if the intensity is the same
                mydict[i1] += 1 # This is to add 1 to the number of times the intensity appears
    return mydict # This is to return the dictionary


def original_histogram(imgName): # This is a function to create the histogram of the original image
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE) # This is to read the image
    dst = cv2.calcHist(img, [0], None, [256], [0,256])  # This is to create the histogram
    # #Grayscale histogram
    plt.figure(figsize=(15,6)) # This is to set the size of the figure
    plt.subplot(1, 2, 1) # This is to set the subplot
    get = plt.hist(img.ravel(), bins = 256) # This is to create the histogram
    plt.title('Original Histogram from OpenCV') # This is to set the title
    return img
    # plt.show()

if rank == 0:
    sdata = original_histogram(imageName) # This is to run the function to create the histogram of the original image
    print('shape of original image is', sdata.shape) # This is to print the shape of the original image
    sdata = np.array_split(sdata, size) # This is to split the image into chunks

else:
    sdata = None

data = comm.scatter(sdata, root=0) # This is to scatter the image chunks to the other processes
data = [l.tolist() for l in data] # This is to convert the image chunks to a list
# print('i am rank {} and i have {}'.format(rank, data))
flat_list = [item for sublist in data for item in sublist] # This is to flatten the list
mycounts = Counter(flat_list) # This is to count the number of times each intensity appears in the image
# mycounts = mycounter(flat_list) # This is to count the number of times each intensity appears in the image
# print(mycounts)


for i in range(len(bucket)): # This is to iterate through the intensities
    if i in mycounts: # This is to check if the intensity appears in the image
        bucket[i] = mycounts[i] # This is to add the number of times the intensity appears to the bucket
# print(bucket)

recv = comm.reduce(bucket) # This is to reduce the bucket to the root process
# print('recv is ', recv)

if recv is not None:
    # print(len(recv))
    x =np.arange(256) # This is to create the x-axis
    # print('i am rank {} and the recvb is {}'.format(rank, recv))
    plt.subplot(1, 2, 2) # This is to set the subplot
    plt.bar(x, recv, color = 'r')  # This is to create the histogram
    plt.title('Histogram from Collective Communication') # This is to set the title
    et = MPI.Wtime() # This is to get the end time
    print('running time is {} for size {}'.format(round(et-st, 5), size)) # This is to print the running time
    plt.show() # This is to show the histogram



