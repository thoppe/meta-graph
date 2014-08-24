import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector, grab_all
from src.invariants import graph_tool_representation
import src.invariants as invar

import graph_tool
import graph_tool.topology
import graph_tool.spectral

desc   = "Creates the connected simple edge meta-graph of order N"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N',type=int,default=4,
                    help="graph order")
parser.add_argument('--draw', default=False, action='store_true')
parser.add_argument('--clear', default=False, action='store_true',
                    help="Clears this value from the database")

cargs = vars(parser.parse_args())
N = cargs["N"]

upper_idx = np.triu_indices(N)

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn  = load_graph_database(N)
sconn = load_graph_database(N,special=True)

meta_conn = sqlite3.connect("simple_meta.db")

# Connect to the database
with open("template_meta.sql") as FIN:
    script = FIN.read()
    meta_conn.executescript(script)
    meta_conn.commit()


if cargs["clear"]:
    cmd_clear = '''DELETE FROM metagraph WHERE meta_n = (?)'''
    logging.warning("Clearing metagraph values {}".format(N))
    meta_conn.execute(cmd_clear, (N,))
    meta_conn.commit()

    meta_conn.execute("VACUUM")
    meta_conn.commit()

    cmd_clear_complete = '''DELETE FROM computed WHERE meta_n = (?)'''
    meta_conn.execute(cmd_clear_complete, (N,))
    exit()

cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(meta_conn,cmd_check_complete)
if N in complete_n:
    msg = "meta {} has already been computed, exiting".format(N)
    raise ValueError(msg)

    
# Make a mapping of all the graph id's
logging.info("Loading the graph adj information")
ADJ = dict(grab_all(conn, '''SELECT graph_id,adj FROM graph'''))

# Grab all the Laplacian polynomials
logging.info("Loading the Laplacian database")
single_grab = '''
SELECT x_degree,coeff FROM laplacian_polynomial
WHERE graph_id = (?) ORDER BY x_degree,coeff'''
LPOLY = collections.defaultdict(list)

for gid in ADJ:
    L = tuple(grab_all(sconn, single_grab, (gid,)))   
    LPOLY[L].append(gid)

assert(len(LPOLY) == len(ADJ))


__upper_matrix_index = np.triu_indices(N)

def is_connected(g):
    component_size = graph_tool.topology.label_components(g)[1]
    return len(component_size)==1

def iso_difference_in_set(g, iso_set):
    ''' Find an isomorphism in a list and return it, otherwise return True'''
    for h in iso_set:
        if graph_tool.topology.isomorphism(g,h):
            return False,h
    return True,None

def connected_cut_iterator(g):
    ''' Iterates over all graphs with one less edge that are still connected '''
    for edge in g.edges():

        gx = g.copy()
        gx.remove_edge(edge)

        # Check if connected
        if is_connected(gx):
            yield gx
    

def find_iso_set_from_cut(g):
    # For every edge in the current graph,
    # find isomorphicly distinct set of edge removals and their cardinality

    iso_set = collections.Counter()

    for h in connected_cut_iterator(g):
       
        is_unique, h_iso = iso_difference_in_set(h, iso_set)
        if is_unique: 
            iso_set[h] += 1
        else: 
            iso_set[h_iso] += 1

    return iso_set

def compute_valid_cuts((e0, target_adj)):
    # Determine the valid iso_set and compute the invariant (Laplacian)
    g = graph_tool_representation(target_adj,N=N)
    iso_set = find_iso_set_from_cut(g)

    #laplacian_set = collections.Counter()
    laplacian_map = dict()

    HL = []
    for h in iso_set:
        A = graph_tool.spectral.adjacency(h)
        laplacian_map[h] = invar.special_laplacian_polynomial(A,N=N)

    return e0, iso_set, laplacian_map

def possible_laplacian_match(L):
    return [ (k,ADJ[k]) for k in  LPOLY[L]]

def identify_match_from_adj(g, match_set):
    # Sanity check here, remove comments for speed
    #if len(match_set) == 1:
    #    return match_set[0][0]

    for idx,adj in match_set:
        h = graph_tool_representation(adj,N=N)
        if graph_tool.topology.isomorphism(g,h):
            return idx

    raise ValueError("should find match already")

def process_lap_poly((e0, iso_set,L_MAP)):

    L_GRAPH_MAP = {}
    for h,L in L_MAP.iteritems():
        match_set = possible_laplacian_match(L)
        L_GRAPH_MAP[h] = match_set
    return (e0, iso_set, L_GRAPH_MAP)

def process_match_set((e0,iso_set,L_GRAPH_MAP)):

    E1_weights = {}
    for h,match_set in L_GRAPH_MAP.iteritems():
        e1 = identify_match_from_adj(h,match_set)
        E1_weights[e1] = iso_set[h] 
        
    return (e0,E1_weights)

def record_E1_set((e0,E1_weights)):

    cmd_insert = '''INSERT INTO metagraph VALUES (?,?,?,?)'''

    def edge_insert_itr():
        for e1 in E1_weights:
            yield (N,e0,e1,E1_weights[e1])
    
    meta_conn.executemany(cmd_insert, edge_insert_itr())
    
    


ADJ_iter = ADJ.iteritems()

from multi_chain import multi_Manager
MULTI_TASKS  = [compute_valid_cuts,process_match_set]
SERIAL_TASKS = [process_lap_poly,record_E1_set]
M = multi_Manager(ADJ_iter, MULTI_TASKS, SERIAL_TASKS)
M.run()

cmd_mark_complete = '''INSERT INTO computed VALUES (?)'''
meta_conn.execute(cmd_mark_complete,(N,))
meta_conn.commit()

print "DONE?"
exit()


cmd_check  = '''SELECT e0 FROM metagraph WHERE meta_n={} AND e0={} LIMIT 1'''
compute_size = len(ADJ)


def process_adj((i,target_adj)):

    if i%1000==0:
        print "Starting meta_{}, edge {}".format(N,i)
    new_edges = compute_meta_edges(i,target_adj)
    return new_edges


exit()


import multiprocessing
P = multiprocessing.Pool()

manager = multiprocessing.Manager()
LD = manager.dict(LPOLY)

if N not in complete_n:
    logging.info("Starting the computation for meta_{}".format(N))

    #items = ((gid, adj,LD) for gid,adj in ADJ.items())
    #for v in items:
    #    print process_adj(v)
    #exit()
    items = ADJ.items()

    sol = P.imap(process_adj,items,chunksize=5)
    for k,new_edges in enumerate(sol):
        meta_conn.executemany(cmd_insert, new_edges)


P.close()
P.join()
    
meta_conn.commit()
cmd_mark_complete = '''INSERT INTO computed VALUES (?)'''
meta_conn.execute(cmd_mark_complete, (N,))
meta_conn.commit()

if not cargs["draw"]:
    exit()


from scipy.sparse import *
import graph_tool.draw

print "Building graph-tool representation"

cmd_select = '''SELECT e0,e1 FROM metagraph WHERE meta_n={}'''

m = graph_tool.Graph(directed=False)
m.add_vertex(len(ADJ))

cursor = meta_conn.execute(cmd_select.format(N))
while cursor:
    block = cursor.fetchmany(1000)
    if not block: break
    for edge in block:
        i,j = edge
        m.add_edge(i-1,j-1)

f_png = "figures/meta_simple_{}.png".format(N)
logging.info("Saving %s"%f_png)

print "Saving"
f_save = "reps/meta_{}.gml".format(N)
m.save(f_save)

print "Drawing"
graph_tool.draw.graphviz_draw(m,layout="dot",
                              output=f_png,
                              vsize=.4,penwidth=4,size=(30,30))

#import pylab as plt
#plt.show()




