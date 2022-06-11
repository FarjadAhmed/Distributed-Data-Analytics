from mpi4py import MPI
import numpy as np
import math
import time
comm = MPI.COMM_WORLD
P = comm.size
rank = comm.rank
root = 0
n = 10**3

st = MPI.Wtime()
def generate_vec(n): 
    vecA = np.random.randint(6, size=(n))
    return vecA

def NsendAll(data):
    if data is not None:
        print('NsendAll called')
        for i in range(1, P):
            comm.send( data, dest = i)
    else:
        get_data = comm.recv( source = root)
        # print('data received is:', get_data, 'and tag is', rank)
    comm.barrier()

def EsendAll(data):
    if data is not None:
        print('EsendAll called')
        # print('i am here and rank is', rank)
        destA = 2 * rank + 1
        destB = 2 * rank + 2
        if destA < P:
            comm.send(data, destA)
        if destB < P:
            comm.send(data, destB)
    else:
        recvProc = int((rank - 1)/2)
        gdata = comm.recv(source = recvProc)
        destA = 2 * rank + 1
        destB = 2 * rank + 2
        if destA < P:
            comm.send(gdata, destA)
        if destB < P:
            comm.send(gdata, destB)
    comm.barrier()


if rank == 0:
    sdata = generate_vec(n)
else: 
    sdata = None

# NsendAll(sdata)
EsendAll(sdata)

et = MPI.Wtime()
print('running time: ', round(et-st, 6))

