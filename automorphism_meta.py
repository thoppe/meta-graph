import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector, grab_scalar
from src.invariants import convert_to_numpy, graph_tool_representation

import graph_tool
import graph_tool.topology

desc   = "Calculate the automorphism group for the metagraph"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('n',type=int,default=4,
                    help="graph order")

cargs = vars(parser.parse_args())
n = cargs["n"]
upper_idx = np.triu_indices(n)

import numpy as np
import ast,itertools, os
import subprocess
import networkx as nx
import graph_tool.topology
import graph_tool.draw

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn = load_graph_database(cargs["n"])

meta_conn = sqlite3.connect("simple_meta.db")

cmd_edges = '''SELECT count(*) FROM metagraph WHERE meta_n={}'''
edge_n = grab_scalar(meta_conn,cmd_edges.format(n))

#cmd_vertex = '''SELECT count(*) FROM metagraph WHERE meta_n={} GROUP BY e0'''
#vertex_n = grab_scalar(meta_conn,cmd_vertex.format(n))

cmd_select = '''SELECT e0,e1 FROM metagraph WHERE meta_n={}'''

cursor = meta_conn.execute(cmd_select.format(n))
s = []

vertex_n = set()
while cursor:
    block = cursor.fetchmany(1000)
    if not block: break
    for j,i in block:
        vertex_n.add(i)
        vertex_n.add(j)
        s.append("e {} {}".format(i+1,j+1))

s = ["p edge {} {}".format(len(vertex_n),edge_n)] + s
s_echo = '"%s"'%('\n'.join(s))
cmd = "echo %s | src/bliss/bliss" % s_echo

proc = subprocess.Popen([cmd],stdout=subprocess.PIPE,shell=True)
for line in proc.stdout:
    if "|Aut|" in line:
        print n, edge_n, line.split()[-1]
