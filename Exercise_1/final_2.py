from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
root = 0
n=10**4
st = MPI.Wtime()

def multiplyV(A,B):
    result = np.dot(B,A)
    return result
def generate_mat(n):
    np.random.seed(0) # seed set to have a determinable solutiohn
    matA = np.random.randint(6, size=(n))
    matB = np.random.randint(5, size=(n,n))
    C = np.dot(matA, matB)
    # print('Result should be', C)
    return matA, matB
if rank == root:    # root process
    result = np.array([]) # create an empty array
    out =[] # create an empty array
    A, B = generate_mat(n) # generate matrix
    transpose= np.transpose(B) # transpose matrix
    Bsplit = np.vsplit(transpose, size) # split matrix B into chunks
    C = np.dot(A, B) # calculate matrix C
    for i in range(size): # send A to each process
        if i != root:  # except root
            comm.send(A, i) # send A to each process
            comm.send(Bsplit[i], dest=i)    # send B to each process
    resultChunks = multiplyV(A, Bsplit[0]) # calculate the first resultant
    # print(resultChunks)
    result = np.append(result, resultChunks) # append result
    for z in range(size): # receive result from process
        if z != root: # except root
            received_resultChunks = comm.recv(source=z) # receive result from process
            result = np.append(result, received_resultChunks) # append result
    # print('Output',result)
else:
    Achunk = comm.recv(source=root) # receive A from root
    Bchunk = comm.recv(source=root) # receive B from root
    product = multiplyV(Achunk, Bchunk) # calculate product
    # print(product)
    comm.send(product, dest=root) # send result to root

et = MPI.Wtime()
actual_time = et-st
print('execution time', actual_time) # print execution time