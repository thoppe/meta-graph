import sqlite3
import argparse
import graph_tool
import graph_tool.draw

from src.helper_functions import load_graph_database, grab_scalar
from src.helper_functions import grab_vector, select_itr

desc = "Draws the connected simple edge meta-graph of order N"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N', type=int, default=4,
                    help="graph order")

cargs = vars(parser.parse_args())
N = cargs["N"]

# Connect to the meta database and template it if needed
f_meta_conn = "meta_db/simple_connected_{}.db".format(N)
meta_conn = sqlite3.connect(f_meta_conn)

# Check if values have been computed, if not, exit early
cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(meta_conn, cmd_check_complete)
if N not in complete_n:
    msg = "meta {} has not been computed, exiting".format(N)
    raise ValueError(msg)

# Count the vertices
cmd_search = '''
SELECT COUNT(DISTINCT e0) FROM metagraph WHERE meta_n={}'''.format(N)
vertex_n = grab_scalar(meta_conn, cmd_search)

cmd_select = '''
SELECT e0,e1 FROM metagraph
WHERE meta_n={} AND direction=0'''.format(N)

m = graph_tool.Graph(directed=False)
m.add_vertex(vertex_n)

# Draw the undirected case, direction==0
for (i, j) in select_itr(meta_conn, cmd_select):
    m.add_edge(i - 1, j - 1)

f_save = "reps/meta_{}.gml".format(N)
print("Saving {}".format(f_save))
m.save(f_save)

f_png = "figures/meta_simple_{}.png".format(N)
print("Drawing {}".format(f_png))
graph_tool.draw.graphviz_draw(m, layout="dot",
                              output=f_png,
                              vsize=.4, penwidth=4, size=(30, 30))
