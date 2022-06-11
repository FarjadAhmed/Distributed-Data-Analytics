from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
root = 0
n=10**4
st = MPI.Wtime()
def generate_mat(n):
    np.random.seed(0) # seed set to have a determinable solutiohn
    matA = np.random.randint(6, size=(n,n)) # nxn matrix
    matB = np.random.randint(5, size=(n,n)) # nxn matrix
    # print(matA.shape) # print shape of matrix
    return matA, matB # return matrix
if rank == root:
    output = np.array([]) # create an empty array
    A, B = generate_mat(n) # generate matrix
    C = np.add(A,B) # add matrix
    # print('output should be', C)
    Asplit = np.array_split(A, size) # split matrix A into chunks
    Bsplit = np.array_split(B, size) # split matrix B into chunks
    # Send each process their chunk
    for i in range(size):
        if i != root:
            comm.send(Asplit[i], dest=i) # send A to each process
            comm.send(Bsplit[i], dest=i) # send B to each process
    outputChunks = np.add(Asplit[root], Bsplit[root]) # add A and B
    output = np.append(output, outputChunks) # append output
# This conditional caters to the output capture and printing
if rank == root:
    for z in range(size): # for each process
        if z != root: # if process is not root
            received_outputChunks = comm.recv(source=z) # receive output from process
            output = np.append(output, received_outputChunks) # append output
    # print('Output',output.reshape(n,n)) # print output
    # et = MPI.Wtime() # end time
    # print('execution time', et-st) # print execution time
else:
    Achunk = comm.recv(source=root) # receive A from root
    Bchunk = comm.recv(source=root) # receive B from root
    summation = np.add(Achunk, Bchunk) # add A and B
    comm.send(summation, dest=root) # send output to root
et = MPI.Wtime()
actual_time = et-st
print('execution time', actual_time) # print execution time