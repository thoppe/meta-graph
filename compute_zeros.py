import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.spectral, graph_tool.draw
import scipy
from src.helper_functions import select_itr, grab_scalar, grab_all
import src.invariants as invar

desc   = "Computes zeros of the simple edge meta-graph of order n"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N',type=int,default=4,
                    help="graph order")
cargs = vars(parser.parse_args())
N = cargs["N"]

# Connect to the meta database
f_meta_conn = "meta_db/simple_connected_{}.db".format(N)
meta_conn = sqlite3.connect(f_meta_conn)

cmd_count = "SELECT COUNT(*) FROM graph"
vertex_n = grab_scalar(meta_conn, cmd_count)
print vertex_n

cmd_count = "SELECT COUNT(*) FROM metagraph"
edge_n = grab_scalar(meta_conn, cmd_count)

#from scipy.sparse import dok_matrix
#A = dok_matrix((vertex_n, vertex_n))
A = np.zeros((vertex_n,vertex_n))#,dtype=np.complex128)

#g = gt.Graph(directed=True)
#g.add_vertex(vertex_n)
e_itr = select_itr(meta_conn,"SELECT e0,e1,weight FROM metagraph")
for (e0,e1,w) in e_itr:
    for _ in xrange(w):
        i,j = e0-1,e1-1
        A[i,j] += 1

L0 = np.zeros(A.shape)
L1 = np.zeros(A.shape)
np.fill_diagonal(L0, A.sum(axis=0))
np.fill_diagonal(L1, A.sum(axis=1))
L0 -= A
L1 -= A

#print scipy.linalg.eigvals(L0)
#print scipy.linalg.eigvals(L1)
#exit()



'''
seq_of_zeros = scipy.linalg.eigvals(A)
epsilon = .00001
seq_of_zeros[np.abs(seq_of_zeros.imag)<epsilon].imag = 0
seq_of_zeros[np.abs(seq_of_zeros.real)<epsilon].real = 0

#import numpy.core.numeric as NX
#convolve = NX.convolve

from scipy.signal import convolve
from scipy.signal import fftconvolve as convolve

terms = np.ones((1.0,))
for k in range(len(seq_of_zeros)):
    terms = convolve(terms, np.array([1, -seq_of_zeros[k]]), mode='full')

print terms
#roots = np.roots(terms)
#print roots
exit()
#exit()
#p = np.poly(A)
#print p
#roots = np.roots(p)
#print roots
'''
import pylab as plt
import seaborn as sns

def P_w(r,Z_b,beta):
    return (1.0/Z_b)* ((r+r**2)**beta/((1+r+r**2)**(1+(3./2)*beta)) )

# Unfolded 
def GOE(s):
    return (np.pi/2)*s*np.exp(-(np.pi/4)*s**2)

def GUE(s):
    return (32/np.pi**2)*s**2*np.exp(-(4/np.pi)*s**2)

#def GUE(r):
#    return P_w(r, (4/81.)*(np.pi/np.sqrt(3.0)), 2.0)

#def GSE(r):
#    return P_w(r, (4/729.)*(np.pi/np.sqrt(3.0)), 4.0)

def POISSON(r):
    return np.exp(-r)

#sns.distplot(roots.real,bins=50,color='g',kde=False)

#roots = scipy.linalg.eigvals(L1)
#sns.distplot(roots.real,bins=50,color='b',kde=False)

sns.set(style="white", palette="muted")
f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=False)
axes = axes.ravel()

roots = scipy.linalg.eigvals(L0
#roots = scipy.linalg.eigvals(A)
sns.distplot(roots.real,bins=50,color='b',kde=False,ax=axes[0])

print "Imag roots", np.abs(roots.imag).sum()
roots = np.sort(roots.real)
diff  = np.diff(roots)
diff /= diff.mean()

#sns.distplot(diff,bins=50,color='b',kde=True,ax=axes[1], 
#             label="Metagraph {} adj spectrum".format(N))
#bins = np.logspace(0,np.log10(3.0),100)-1
bins = np.linspace(0,3,50)
#print scipy.integrate.trapz(diff,bins)
plt.hist(diff,bins=bins,
         histtype="stepfilled", alpha=.5,
         label="Metagraph {} adj spectrum".format(N),
         normed=True)
plt.xlim(xmin=0)

R = np.linspace(0,3,1000)
plt.plot(R, POISSON(R),label="Poisson",color='k')
plt.plot(R, GOE(R),label="GOE")
plt.plot(R, GUE(R),label="GUE")
#plt.plot(R, GSE(R),label="GSE")

plt.legend(loc='best')


plt.show()


#plt.title("Roots of the weighted meta-graph Laplacian (in and out) {}".
#          format(N))
#
#plt.savefig("figures/zeros_{}.png".format(N),bbox_inches='tight')
#plt.show()
