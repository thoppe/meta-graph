import numpy as np
import networkx as nx
from pylab import plt

dx, dy = 2.5,3.5

dargs = {
    'with_labels':False,
    'node_color':'#feb24c',
'width':6
}

center = np.array([(0,1),(1,1),(1,0),(0,0)])-[.5,.5]

g =  nx.Graph()
g.add_edges_from( ((1,2),(2,4),(4,3),(3,1),(1,4),(2,3)) )

G   = [g,]
POS = [center,]

g =  nx.Graph()
g.add_edges_from( ((1,2),(4,3),(3,1),(1,4),(2,3)) )
G.append(g)
POS.append( center - [0,dy] )


g =  nx.Graph()
g.add_edges_from( ((4,3),(3,1),(1,4),(2,3)) )
G.append(g)
POS.append( center - [dx,2*dy] )

g =  nx.Graph()
g.add_edges_from( ((1,2),(4,3),(1,4),(2,3)) )
G.append(g)
POS.append( center - [-dx,2*dy] )

########

g =  nx.Graph()
g.add_edges_from( ((4,3),(3,1),(2,3)) )
G.append(g)
POS.append( center - [dx,3*dy] )

g =  nx.Graph()
g.add_edges_from( ((4,3),(1,4),(2,3)) )
G.append(g)
POS.append( center - [-dx,3*dy] )

###################

# Build the meta
M = nx.Graph()
pos = {}
pos[1] = [0,0]
pos[2] = [0,-dy]
pos[3] = [-dx,-2*dy]
pos[4] = [dx,-2*dy]
pos[5] = [-dx,-3*dy]
pos[6] = [dx,-3*dy]
M.add_edges_from([ (1,2),(2,3),(2,4),(4,6),(3,6),(3,5) ])
nx.draw_networkx_edges(M,pos,color='r',style='--',width=3,zorder=-10,alpha=.5)

nx.draw_networkx_nodes(M,pos,width=6,node_color='white',
        with_labels=False,node_size=4500,alpha=.3,zorder=10)

###################

for g,pos in zip(G,POS):
    pos_map = dict(zip(range(1,5),pos))
    nx.draw_networkx_nodes(g,pos_map,zorder=20,**dargs)
    nx.draw_networkx_edges(g,pos_map,zorder=20,**dargs)

########

plt.axis('off')
plt.axis('equal')
plt.savefig("figures/example_4.png",bbox_inches='tight',bbox_inches=0)
plt.show()

#print k4
