import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector, grab_all
from src.invariants import convert_to_numpy, graph_tool_representation
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

save_conn = sqlite3.connect("simple_meta.db")

# Connect to the database
with open("template_meta.sql") as FIN:
    script = FIN.read()
    save_conn.executescript(script)
    save_conn.commit()


if cargs["clear"]:
    cmd_clear = '''DELETE FROM metagraph WHERE meta_n = (?)'''
    logging.warning("Clearing metagraph values {}".format(N))
    save_conn.execute(cmd_clear, (N,))
    save_conn.commit()

    save_conn.execute("VACUUM")
    save_conn.commit()

    cmd_clear_complete = '''DELETE FROM computed WHERE meta_n = (?)'''
    save_conn.execute(cmd_clear_complete, (N,))
    exit()

# Make a mapping of all the graph id's
logging.info("Grabbing the graph adj information")
ADJ = dict(grab_all(conn, '''SELECT graph_id,adj FROM graph'''))

# Grab all the Laplacian polynomials
logging.info("Generating the Laplacian database")
single_grab = '''
SELECT x_degree,coeff FROM laplacian_polynomial
WHERE graph_id = (?) ORDER BY x_degree,coeff'''
LPOLY = collections.defaultdict(list)

'''
for gid,adj in ADJ.items():
    g  = graph_tool_representation(adj,N=N)
    gA = graph_tool.spectral.adjacency(g).toarray().astype(int)
    gadj = convert_numpy_to_adj(gA)
    invar.viz_graph(g)
exit()
'''

for gid in ADJ:
    L = tuple(grab_all(sconn, single_grab, (gid,)))   
    LPOLY[L].append(gid)


__upper_matrix_index = np.triu_indices(N)

def convert_numpy_to_adj(A):
    # The string representation of the upper triangular adj matrix
    au = ''.join(map(str, A[__upper_matrix_index]))

    # Convert the binary string to an int
    int_index = int(au, 2)

    return int_index

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

def compute_valid_cuts(target_adj):
    # Determine the valid iso_set and compute the invariant (Laplacian)
    g = graph_tool_representation(target_adj,N=N)
    iso_set = find_iso_set_from_cut(g)

    #laplacian_set = collections.Counter()
    laplacian_map = dict()

    HL = []
    for h in iso_set:
        A = graph_tool.spectral.adjacency(h)
        laplacian_map[h] = invar.special_laplacian_polynomial(A,N=N)

    return iso_set, laplacian_map

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

def record_meta_edge(e0,e1,weight):
    print "metaedge_{}: {} * ({},{})".format(N,weight,e0,e1)

for idx,adj in ADJ.items():

    iso_set, L_MAP = compute_valid_cuts(adj)
    for h,L in L_MAP.items():
        match_set = possible_laplacian_match(L)
        match_idx = identify_match_from_adj(h,match_set)
        weight = iso_set[h]
        record_meta_edge(idx,match_idx,weight)

exit()

'''
        print hL
        exit()

        items = [(k,ADJ[k]) for k in LPOLY[hL]]

        isomorphism_check_counter.append(len(items))
        j = find_iso_match(h, items)

        new_edges.append( (N,i,j) )

    #print " + number of isomorphism checks to complete", sum(isomorphism_check_counter)
    return new_edges
'''




cmd_insert = '''INSERT INTO metagraph VALUES (?,?,?)'''
cmd_check  = '''SELECT e0 FROM metagraph WHERE meta_n={} AND e0={} LIMIT 1'''
compute_size = len(ADJ)

cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(save_conn,cmd_check_complete)

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
        save_conn.executemany(cmd_insert, new_edges)


P.close()
P.join()
    
save_conn.commit()
cmd_mark_complete = '''INSERT INTO computed VALUES (?)'''
save_conn.execute(cmd_mark_complete, (N,))
save_conn.commit()

if not cargs["draw"]:
    exit()


from scipy.sparse import *
import graph_tool.draw

print "Building graph-tool representation"

cmd_select = '''SELECT e0,e1 FROM metagraph WHERE meta_n={}'''

m = graph_tool.Graph(directed=False)
m.add_vertex(len(ADJ))

cursor = save_conn.execute(cmd_select.format(N))
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




