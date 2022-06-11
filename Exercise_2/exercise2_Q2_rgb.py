from operator import invert
import re
from typing import Counter
from mpi4py import MPI
import numpy as np
import cv2
from itertools import groupby
import matplotlib.pyplot as plt
st = MPI.Wtime()
bucket = np.zeros(256)
imageName = 'bigpic.png'
# imageName = 'smallrgb.png'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0
channels = 3 # This is to set the number of channels
 # This is to create an array of arrays of zeros
bucket_arrays = np.array([np.zeros(256), np.zeros(256), np.zeros(256)])
num_of_intensities = 256 * channels


def mycounter(lst): # This is a function to count the number of times each intensity appears in the image
    mydict = {} # This is a dictionary to store the number of times each intensity appears
    for i1 in lst: # This is a loop to iterate through the intensities 
        mydict[i1] = 0 # This is to initialize the dictionary with 0
        for i2 in lst: # This is a loop to iterate through the intensities
            if i1 == i2: # This is to check if the intensity is the same
                mydict[i1] += 1 # This is to add 1 to the number of times the intensity appears
    return mydict # This is to return the dictionary

def original_histogram(imgName): # This is a function to create the histogram of the original image
    img = cv2.imread(imgName) # This is to read the image
    img = img[:3900, :5870, :] # This is to crop the image
    print('Restructured dimensions of original image are', img.shape) # This is to print the shape of the original image
    color = ('b','g','r') # This is to set the color channels
    plt.figure(figsize=(10,6)) # This is to set the size of the figure
    plt.subplot(1, 2, 1) # This is to set the subplot
    plt.title('Original Image Histogram') # This is to set the title
    for i,col in enumerate(color): # This is to iterate through the color channels
        histr = cv2.calcHist([img],[i],None,[256],[0,256]) # This is to create the histogram
        plt.plot(histr,color = col) # This is to plot the histogram
        plt.xlim([0,256]) # This is to set the x-axis limits
    # plt.show()
    return img

if rank == 0:
    img = original_histogram(imageName) 
    img = img[:3900, :5870, :] # Restructuring the image to make it easier to split
    imgb = img[:,:,0] # This is to set the blue channel
    imgg = img[:,:,1] # This is to set the green channel
    imgr = img[:,:,2] # This is to set the red channel
    stacked = np.vstack((imgb, imgg, imgr)) # Stacking the image channels
    # splitCal = int(img.shape[1]/size)
    # print('splitCal is', splitCal)
    # since we are doing hsplit, 5 works for an image with 50 columns
    sdata = np.hsplit(stacked, size) # Splitting the image into chunks
    # splits = np.split(stacked, size) 
    # comments for an image of shape 40x50x3
    # print('splits are of size', splits[0].shape)
    # rgb channels stacked, so 40 x 50 per each channel gives a stacked matrix of 120x50
    # after split by columns we get 5 matrices of 120x10 length, here 120 is contains 3 chunks each of size 40 for each channel 
else:
    sdata = None

data = comm.scatter(sdata, root=0) # Scatter the data to all the processes
vsplit = np.vsplit(data,channels) # Splitting the data into channels

for chans in range(channels):  # Looping through the channels
    v = vsplit[chans] # Getting the channel
    s1 = [l.tolist() for l in v] #list of no arrays to list of lists
    flat_s2 = [item for sublist in s1 for item in sublist] #list of lists to list
    mycounts = Counter(flat_s2) # Counting the number of times each intensity occurs
    # mycounts = mycounter(flat_s2) # Counting the number of times each intensity occurs
    for i in range(len(bucket)): # Looping through the intensities
        if i in mycounts: # If the intensity is present in the list
            bucket_arrays[chans][i] = mycounts[i] # Assigning the count to the bucket

recv = comm.reduce(bucket_arrays) # Reduce the buckets to the root

if recv is not None: 
    x =np.arange(256)  # x axis
    colorlst = ['b','g','r'] # color list
    plt.subplot(1, 2, 2) # subplot
    plt.title('Histogram from Collective Communication') # title
    for freq_arrays in range(channels): # Looping through the channels
        plt.bar(x, recv[freq_arrays], color = colorlst[freq_arrays]) # plotting the histogram
        plt.xlim([0,256]) # x axis limits
    et = MPI.Wtime() # End time
    print('running time is {} for size {}'.format(round(et-st, 5), size)) # print the time taken
    plt.show() # show the plot



