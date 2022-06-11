from mpi4py import MPI
import numpy as np
comm=MPI .COMM_WORLD
rank = comm.rank
size= comm.Get_size()
n=10**2
st=MPI.Wtime()

def generate_mat(n):
    np.random.seed(0) # seed set to have a determinable solutiohn
    matA = np.random.randint(6, size=(n,n)) # nxn matrix
    matB = np.random.randint(5, size=(n,n)) # nxn matrix
    return matA, matB

if(rank==0):
  A, B =  generate_mat(n)
  A_Chunks = np.vsplit(A, size)

else:
  A=None
  B=None
  C=None
  A_Chunks=None
recvA= comm.scatter(A_Chunks,root=0)
bcast_matB= comm.bcast(B,root=0)
product=np.dot(recvA,bcast_matB)
output = comm.gather(product , root=0)

# if rank==0:
#     # print('The result is',output)
    

et = MPI.Wtime()
actual_time = et-st
print('execution time', actual_time) # print execution time

