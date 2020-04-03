#Integrate pi by Int_0_1 (1/(1+x^2))dx = arctan(x)_0_1 = pi/4
from mpi4py import MPI  
import sys
import numpy as np

def f(x):
    return 4.0/(1.0+pow(x,2))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    n = int(input('enter the number of intervals: '))
    if n < 1:
        sys.exit("error number of processes is less than 1")
elif rank != 0:
    n = None
n = comm.bcast(n,root=0)


h = 1.0/float(n)
sum = np.array(0.0,'d')
i = rank + 1
while i <= n:
    x = h * (float(i) - 0.5)
    # print('rank: ',rank,' i: ',i,' x:',x)
    sum += np.array(f(x),'d')
    # print('rank: ',rank,' sum: ',sum)
    i+= nprocs

mypi = h*sum
pi = np.array(0.0,'d')
comm.Reduce(mypi,pi,op=MPI.SUM,root=0)

if rank == 0:
    print('My value of pi is: ',pi)
    print('The error is: ',(pi - np.pi))