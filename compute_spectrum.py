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

# Connect to the meta database
f_meta_conn = "meta_db/simple_connected_{}.db".format(N)
meta_conn = sqlite3.connect(f_meta_conn)

cmd_count = "SELECT COUNT(*) FROM graph"
vertex_n = grab_scalar(meta_conn, cmd_count)
print vertex_n

cmd_count = "SELECT COUNT(*) FROM metagraph"
edge_n = grab_scalar(meta_conn, cmd_count)

g = gt.Graph(directed=True)
g.add_vertex(vertex_n)
e_itr = select_itr(meta_conn,"SELECT e0,e1,weight FROM metagraph")
for (e0,e1,w) in e_itr:
    for _ in xrange(w):
        g.add_edge(e0-1,e1-1)

g_reduced = gt.Graph(directed=False)
g_reduced.add_vertex(vertex_n)
e_itr = select_itr(meta_conn,'''SELECT e0,e1 
FROM metagraph WHERE direction =0''')
for (e0,e1) in e_itr:
    g_reduced.add_edge(e0-1,e1-1)

T = gt.spectral.transition(g).T

L,V = scipy.linalg.eig(T.todense())
idx = np.argsort(L.real)[::-1]
L = L[idx]
V = V[:,idx]

def draw_spectral(g, v_weight,output=None):
    vprop = g.new_vertex_property("double")
    for v in g.vertices():
        vprop[v] = v_weight[int(v)]

    draw = graph_tool.draw.graphviz_draw
    draw(g,layout="dot",
         vcolor=vprop,
         vsize=.5,penwidth=2,size=(30,30),
         output=output)

for k in xrange(4):
    v = V.T[k]
    v /= np.linalg.norm(v)
    if sum(v)<0: v *= -1
    v /= np.abs(v).max()
    print v
   
    f_png = "figures/modes_N_{}_k_{}.png".format(N,k)

    draw_spectral(g_reduced, v,output=f_png)

exit()

