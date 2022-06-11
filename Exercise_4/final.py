import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
np.set_printoptions(suppress=True)
from code import interact
from pkgutil import get_data
from mpi4py import MPI
import numpy as np
import pandas as pd
np.random.seed(0)
comm = MPI.COMM_WORLD # create a communicator
size = comm.size # get the size of the cluster
rank = comm.rank # get the rank of the process
root = 0 # root process
st = MPI.Wtime()  # start time
file = 'virus_dataset.csv' # file name
learning_rate=0.1 # learning rate
epochs=20 # number of epochs
diminish_lr=1.01 # learning rate
k=100 # number of samples



def meanSquaredError(y_test,y_pred): # Mean Squared Error
    MSE = np.mean(np.square(np.subtract(y_test,y_pred))) # manually computing MSE
    return MSE

def read_file(file): # reading file
    df = pd.read_csv(file) # reading csv file
    return df

def split_data(X, y, training_size): # Splitting data
    total_size = X.shape[0] # total size of the data
    ind = total_size  = int(training_size * total_size) # index of the test data
    X_train = X[:ind,:] # training data
    y_train = y[:ind] # training data
    X_test =  X[ind:,:] # test data
    y_test = y[ind:] # test data
    return X_train, X_test, y_train, y_test

def get_data(): # reading file
    data = read_file(file) # reading csv file
    data = data.sample(frac=1) # shuffling the data
    data = data.to_numpy() # converting data to numpy array
    # np.random.shuffle(data) # shuffling the data
    Y = data[:, -1] # target variable
    X = data[:,:(data.shape[1]-1)] # features
    x_train,x_test,y_train,y_test=split_data(X,Y, 0.8) # splitting data
    # print("X Shape: ",X.shape) # printing shape of X
    # print("Y Shape: ",Y.shape) # printing shape of Y
    # print("X_Train Shape: ",x_train.shape) # printing shape of X_train
    # print("X_Test Shape: ",x_test.shape) # printing shape of X_test
    # print("Y_Train Shape: ",y_train.shape) # printing shape of Y_train
    # print("Y_Test Shape: ",y_test.shape) # printing shape of Y_test
    # Standardizing data
    x_train = (x_train - np.mean(x_train))/len(x_train) # standardizing X_train
    x_test = (x_test - np.mean(x_test))/len(x_test) # standardizing X_test
    train_data=pd.DataFrame(x_train) # converting X_train to dataframe
    train_data['target'] = y_train # adding target variable to dataframe
    x_test=np.array(x_test) # converting X_test to numpy array
    y_test=np.array(y_test) # converting Y_test to numpy array
    return x_train,x_test,y_train,y_test, train_data

def sklearn_result(x_train, y_train, y_test, x_test): # sklearn result
    # SkLearn SGD classifier 
    clf_ = SGDRegressor(learning_rate= 'adaptive' , alpha=1, max_iter=500, shuffle=False) # creating SGD classifier
    clf_.fit(x_train, y_train) # fitting the classifier
    y_pred_sksgd=clf_.predict(x_test) # predicting the test data
    print('Mean Squared Error from SKlearn :',meanSquaredError(y_test, y_pred_sksgd)) # printing MSE

def initialize_terms(shape): # initializing terms
    w=np.zeros(shape=(1,shape-1)) # initializing w
    # w = np.random.uniform(low = 0.0001, high = 0.005, size=(1,shape-1)) # initializing w
    b = 0 # initializing b
    return w, b

def update_terms(learning_rate, k, x, y, w, b, w_gradient, b_gradient): # updating terms  
    for i in range(k): # looping over k samples
            prediction=np.dot(w,x[i])+b # prediction
            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction)) # updating w_gradient
            b_gradient=b_gradient+(-2)*(y[i]-(prediction)) # updating b_gradient
    w=w-learning_rate*(w_gradient/k) # updating w
    b=b-learning_rate*(b_gradient/k) # updating b
    return w, b  

def get_result(y_test, y_pred): # getting result
    print('Mean Squared Error :', meanSquaredError(y_test, y_pred)) # printing MSE

def SGD(train_data, x_test, y_test, learning_rate, epochs, k, diminish_lr): # SGD
    local_mse = [] # local mse
    timeList = [] # time list
    train_data = pd.DataFrame(train_data) # converting train_data to dataframe
    w, b = initialize_terms(train_data.shape[1]) # initializing terms
    for _ in range(epochs): # looping over epochs
        temp = train_data.sample(k).to_numpy() # sampling k samples
        y = temp[:, -1] # target variable
        x = temp[:,:(temp.shape[1]-1)] # features
        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1)) # initializing w_gradient
        b_gradient=0 # initializing b_gradient
        w, b = update_terms(learning_rate, k, x, y, w, b, w_gradient, b_gradient) # updating terms
        learning_rate=learning_rate/diminish_lr # diminishing learning rate
        y_pred_at_epochs = predict(x_test,w,b)  # predicting the test data
        local_mse.append(mean_squared_error(y_pred_at_epochs, y_test)) # appending local mse
        timeList.append(MPI.Wtime()) # appending time
    return w,b, local_mse, timeList # returning w,b, local_mse, timeList

def predict(x,w,b): # predicting
    y_pred=[] # initializing y_pred
    for i in range(len(x)): # looping over x
        y_pred.append(np.asscalar(np.dot(w,x[i])+b)) # appending y_pred
    return np.array(y_pred) # returning y_pred

if rank == root: # root process
    x_train,x_test,y_train,y_test, train_data = get_data() # getting data
    # sklearn_result(x_train, y_train, y_test, x_test) # sklearn result
    train_data = train_data.to_numpy() # converting train_data to numpy array
    sdata = np.array_split(train_data, size) # splitting data
    sx_test = np.array_split(x_test, size) # splitting x_test
    sy_test = np.array_split(y_test, size) # splitting y_test
    net_time = []
else:
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    train_data = None
    sdata = None
    sx_test = None
    sy_test = None
    net_time = None

sdata = comm.scatter(sdata, root = root) # scattering data
sx_test = comm.scatter(sx_test, root = root)  # scattering x_test
sy_test = comm.scatter(sy_test, root = root) # scattering y_test
# sx_test = comm.bcast(x_test, root = root)
# sy_test = comm.bcast(y_test, root = root)
w,b, mseList, timeList = SGD(sdata,sx_test,sy_test,learning_rate=learning_rate,epochs=epochs,diminish_lr=diminish_lr,k=k) # SGD
times = comm.gather(timeList, root = root) # gathering timeList
mseGather = comm.gather(mseList, root = root) # gathering mseList
w1 = comm.reduce(w, root = root) # reducing w
b1 = comm.reduce(b, root = root) # reducing b

if w1 is not None:
    if b1 is not None:
        w1 = w1/size # dividing w by size
        b1 = b1/size # dividing b by size
        y_pred = predict(x_test,w1,b1) # predicting
        get_result(y_test, y_pred) # getting result

et = MPI.Wtime() # ending time
ttime = et-st # calculating time taken
# print('Time is {} for rank {}: '.format(rank, ttime)) # printing time
totaltime = comm.reduce(ttime, root = root) # reducing time
if totaltime is not None:
    print('Avg Total time is {}'.format(totaltime/size)) # printing avg time

# Graphing is done out of the recorded time, time if for the running of the model
# learning vs epochs
if mseGather is not None:
    plt.figure(figsize=(15,8))
    # plt.subplot(1, 2, 1) # subplot
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('SGD learning vs epochs')
    plt.suptitle('k:{}, size:{}, learning_rate:{}, epochs:{}'.format(k, size, learning_rate, epochs))
    for m in range(len(mseGather)):
        plt.plot(np.arange(len(mseGather[m])),mseGather[m], label='rank:{}'.format(m))
    plt.legend()
    plt.show()

# learning vs time
if times is not None:
    if mseGather is not None:
        # plt.subplot(1, 2, 2) # subplot
        plt.figure(figsize=(15,8))
        plt.xlabel('Time(Seconds)')
        plt.ylabel('MSE')
        plt.title('SGD learning vs time')
        plt.suptitle('k:{}, size:{}, learning_rate:{}'.format(k, size, learning_rate))
        for m in range(len(times)):
            plt.plot(times[m], mseGather[m], label='rank:{}'.format(m))
        plt.legend()
        plt.show()

