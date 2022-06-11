from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
root = 0
n=10**4
st = MPI.Wtime()
def generate_vec(n):
    np.random.seed(0) # seed set to have a determinable solutiohn
    matA = np.random.randint(5, size=(n)) # nxn matrix  
    return matA
if rank == root:
    result = np.array([]) # create an empty array
    A = generate_vec(n) # generate matrix
    # print('avg should be', np.mean(A)) # print average
    Asplit = np.array_split(A, size) # split matrix A into chunks
    # print(Asplit)
    for i in range(1, size):
        comm.send(Asplit[i], dest=i) # send A to each process
    resultChunks = np.mean(Asplit[0]) # calculate the first resultant
    result = np.append(result, resultChunks) # append result
    for z in range(1, size):
            received_resultChunks = comm.recv(source=z) # receive result from process
            result = np.append(result, received_resultChunks) # append result
    # print('average is',np.mean(result)) # print average
    
    
else:
    Achunk = comm.recv(source=root) # receive A from root
    avg = np.mean(Achunk) # calculate average
    comm.send(avg, dest=root) # send average to root
et = MPI.Wtime()
actual_time = et-st
print('execution time', actual_time) # print execution time