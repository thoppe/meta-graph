import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.spectral, graph_tool.draw
import scipy
from src.helper_functions import select_itr, grab_scalar, grab_all
import src.invariants as invar

desc   = "Computes invariants of the simple edge meta-graph of order n"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N',type=int,default=4,
                    help="graph order")
cargs = vars(parser.parse_args())
N = cargs["N"]

# Connect to the database
meta_conn = sqlite3.connect("simple_meta.db")

cmd_count = "SELECT COUNT(*) FROM graph WHERE n=(?)"
vertex_n = grab_scalar(meta_conn, cmd_count, (N,))
print vertex_n
exit()

# Count the vertices
def count_vertex(N):
    VSET = set()
    cmd = '''SELECT e0,e1 FROM metagraph WHERE meta_n = {}'''.format(N)
    for e0,e1 in select_itr(conn,cmd):
        VSET.add(e0)
        VSET.add(e1)
    return len(VSET)

# Grab the edges
def edge_itr(N):
    cmd = '''SELECT e0,e1 FROM metagraph WHERE meta_n = {}'''.format(N)
    for e0,e1 in select_itr(conn,cmd):
        yield e0,e1

def build_networkx(N):
    g = nx.Graph(directed=False)
    for e0,e1 in edge_itr(N):
        g.add_edge(e0,e1)
    return g

def build_graph_tool(N):
    g = gt.Graph(directed=False)
    V = count_vertex(N)
    g.add_vertex(V)
    for e0,e1 in edge_itr(N):
        g.add_edge(e0-1,e1-1)
    return g

def compute_adj_spectrum(g):
    A = gt.spectral.adjacency(g)
    L,V = scipy.linalg.eigh(A.todense())
    idx = np.argsort(L)[::-1]
    L = L[idx]
    return L


def compute_trans_spectrum(g):
    T = gt.spectral.transition(g).T
    L,V = scipy.linalg.eig(T.todense())
    idx = np.argsort(L.real)[::-1]
    L = L[idx]
    V = V[:,idx]
    return L.real, V

def draw_spectral(g, v_weight,output=None):
    vprop = g.new_vertex_property("double")
    for v in g.vertices():
        vprop[v] = v_weight[int(v)]

    draw = graph_tool.draw.graphviz_draw
    draw(g,layout="dot",
         vcolor=vprop,
         vsize=.4,penwidth=4,size=(30,30),
         output=output)



import pylab as plt
import seaborn as sns

for N in xrange(3,9):
    print "Meta_n: ", N
    g = build_graph_tool(N)
    print "Edges: ", g.num_edges()
    print "Vertices: ", g.num_vertices()
    text = r"$N={}$".format(N)

    L,V = compute_trans_spectrum(g)
    X = np.linspace(0,1,L.size)

    for k in range(min(16,len(L))):
        print N,k

        mode = V.T[k]
        mode /= np.abs(mode).sum()
        f_png = "modes_N_{}_k_{}.png".format(N,k)
        draw_spectral(g, mode,output=f_png)

#    plt.plot(X,L,label=text)
#plt.legend(loc="best")
#plt.show()
