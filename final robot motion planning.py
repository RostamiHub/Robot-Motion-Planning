from email.base64mime import header_length
from operator import le
import networkx as nx
import matplotlib.pyplot as plt
# import the math module 
import math
import numpy as np
from matplotlib.patches import Polygon
import heapq

# ------------------

def coordinate_of_initial_point():
    lst = []
    lst = [int(item) for item in input('enter your start point coordinate: put the positions right after each other without any , or space or (): ').split()]
    return lst
def coordinate_of_goal_point():
    lst = []
    lst = [int(item) for item in input('enter your goal point coordinate: put the positions right after each other without any , or space or (): ').split()]
    return lst

def getting_the_neighbors_start():
    lst = []
    lst = [int(item) for item in input('enter your nodes connected to the start point: put the neighbors right after each other without any , or space or (): ').split()]
    return lst
def getting_the_neighbors_goal():
    lst = []
    lst = [int(item) for item in input('enter your nodes connected to the goal point: put the neighbors right after each other without any , or space or (): ').split()]
    return lst
    
# -------------generating our inputs-------

point_start = coordinate_of_initial_point()
point_goal = coordinate_of_goal_point()

nodes_connected_start = getting_the_neighbors_start()
nodes_connected_goal = getting_the_neighbors_goal()


# ------------------

#plotting polygons

#---------------------------
#coord = [[-8,-6], [-3,-6], [-3,-2], [-8,-2]]
#coord.append(coord[0]) 
#repeat the first point to create a 'closed loop' if you want to use plot method you need to add the first coordinate
#xs, ys = zip(*coord) #create lists of x and y values
# plt.plot(xs,ys)
#plt.fill(xs,ys)

#---------------------------- 
fig, ax = plt.subplots(figsize = (10, 10))
def ObstclePlot(coord):
    coord.append(coord[0]) 
    xs, ys = zip(*coord) #create lists of x and y values
    plt.fill(xs,ys)
    
#-------------------

# coord2 = [[0,1],[8,1],[0,7]]
# coord2.append(coord2[0]) #repeat the first point to create a 'closed loop'
# xs2, ys2 = zip(*coord2) #create lists of x and y values
# plt.fill(xs2,ys2) 
# #node zero is the start node and node 11 is the goal node.
# #nodes and their coordinates 
# ----------------------

coorner_convex = [[-8,-6], [-3, -6],[-3,-2],[-8,-2],[-1,-1],[-6,3],[-1,3],[0,1],[0,7],[8,1]]
coorner_non_convex = [[-2,1]]

# ----------ploting obstacles---------
ObstclePlot(coorner_convex[0:4])
ObstclePlot([coorner_convex[4],coorner_non_convex[0],coorner_convex[5],coorner_convex[6]])
ObstclePlot(coorner_convex[7:10])
# print([coorner_convex[4],coorner_non_convex[0],coorner_convex[5],coorner_convex[6]])
#------------------------------------
G=nx.Graph()
i=1

position_list = [point_start, [-8,-6], [-3, -6],[-3,-2],[-8,-2],[-1,-1],[-6,3],[-1,3],[0,1],[0,7],[8,1],point_goal]

#this function finds the distance between two points
def compute_weights(node_pos1,node_pos2):
    length = math.sqrt((node_pos1[0] - node_pos2[0])**2 + (node_pos1[1] - node_pos2[1])**2)
    return float("{:.2f}".format(length))

#location of the graph nodes
for i in range(0,10):
    tmp = i + 1
    G.add_node(tmp,pos=coorner_convex[i])

G.add_node(0,pos=point_start) #goal node (for evey other goal node you should modify this entry)
G.add_node(11,pos=point_goal) #start node (for evey other start node you should modify this entry)

#adding the edges and the corresponding weights
G.add_edge(1,2,weight=compute_weights(position_list[1],position_list[2]))
G.add_edge(2,3,weight=compute_weights(position_list[2],position_list[3]))
G.add_edge(3,4,weight=compute_weights(position_list[3],position_list[4]))
G.add_edge(1,4,weight=compute_weights(position_list[1],position_list[4]))
G.add_edge(3,5,weight=compute_weights(position_list[3],position_list[5]))
G.add_edge(6,5,weight=compute_weights(position_list[6],position_list[5]))
G.add_edge(6,7,weight=compute_weights(position_list[6],position_list[7]))
G.add_edge(5,7,weight=compute_weights(position_list[5],position_list[7]))
G.add_edge(4,6,weight=compute_weights(position_list[4],position_list[6]))
G.add_edge(3,6,weight=compute_weights(position_list[3],position_list[6]))
G.add_edge(2,5,weight=compute_weights(position_list[2],position_list[5]))
G.add_edge(8,10,weight=compute_weights(position_list[8],position_list[10]))
G.add_edge(8,9,weight=compute_weights(position_list[8],position_list[9]))
G.add_edge(10,9,weight=compute_weights(position_list[10],position_list[9]))
G.add_edge(10,9,weight=compute_weights(position_list[10],position_list[9]))
G.add_edge(6,9,weight=compute_weights(position_list[6],position_list[9]))
G.add_edge(7,9,weight=compute_weights(position_list[7],position_list[9]))
G.add_edge(5,8,weight=compute_weights(position_list[5],position_list[8]))
G.add_edge(5,9,weight=compute_weights(position_list[5],position_list[9]))
G.add_edge(7,8,weight=compute_weights(position_list[7],position_list[8]))
G.add_edge(2,10,weight=compute_weights(position_list[2],position_list[10]))
G.add_edge(5,10,weight=compute_weights(position_list[5],position_list[10]))

# for connecting the start and goal point to the their neighbors
for i in nodes_connected_start:
    G.add_edge(0,i,weight=compute_weights(position_list[0],position_list[i]))
for i in nodes_connected_goal:
    G.add_edge(11,i,weight=compute_weights(position_list[0],position_list[i]))

#gets the position of the nodes from the data structure G 
pos=nx.get_node_attributes(G,'pos')

color_map = []
for node in G:
    if node == 0:
        color_map.append('red') #node 0 is the start node
    elif node == 11:
        color_map.append('red')  #node 11 is the goal node
    else:
        color_map.append('blue') #rest of the nodes are the cornors of the obstlces

nx.draw(G,pos,node_size=350,font_size=20,with_labels=True,node_color=color_map)   #draws the graph
labels = nx.get_edge_attributes(G,'weight')   #gets the weights from the data structure G
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels) #prints the weights as lables on the edges

plt.text(point_goal[0],point_goal[1], 'goal', fontsize =30)
plt.text(point_start[0],point_start[1], 'start', fontsize = 30)

# # ------------------

# def coordinate_of_initial_and_goal_point():
#     lst = []
#     lst = [int(item) for item in input('enter your initial point coordinate: ').split()]
#     return lst
      
# point_initial = coordinate_of_initial_and_goal_point()
# point_goal = coordinate_of_initial_and_goal_point()

# # ------------------

# ----adding the initial point to the nodes----


class Graph:
    def __init__(self, v):
        self.v = v
        self.adj = [[] for i in range(v)]
        self.list_nodes = [[float('inf'), []] for i in range(v)]
    
    def define_adj(self, node1, node2, weight):
        self.adj[node1].append([node2,weight])
        self.adj[node2].append([node1,weight])

    def dijsktra(self, source,dest):
        
        tmp = source
        self.list_nodes[tmp][0] = 0
        visited = [False] * self.v
        
        # for i in range(0,11):
        while tmp != dest:
#             print('tmp is', [tmp])
            if visited[tmp] == False:
                visited[tmp] = True
                heap_array = []
                for j in range(len(self.adj[tmp])):
                    if visited[self.adj[tmp][j][0]] == False:
                        cost = self.list_nodes[tmp][0] + self.adj[tmp][j][1]
#                         print('cost is',cost)
                        if cost < self.list_nodes[self.adj[tmp][j][0]][0]:
                            self.list_nodes[self.adj[tmp][j][0]][0] = cost
                            self.list_nodes[self.adj[tmp][j][0]][1] = self.list_nodes[tmp][1] + [tmp]
                        heapq.heappush(heap_array, (self.list_nodes[self.adj[tmp][j][0]][0], self.adj[tmp][j][0]))
                        # print(heap_array)
            # tmp = heap_array[0][1]
            # print('heap is ', heap_array)
            heapq.heapify(heap_array)
            if not heap_array:
                break
            tmp = heap_array[0][1]
        print('your shortest path is',self.list_nodes[dest][1] + [dest])
        print('weight of the path is',self.list_nodes[dest][0])
        return self.list_nodes[dest][1] + [dest]
        

d = Graph(12)

d.define_adj(1,2,weight=compute_weights(position_list[1],position_list[2]))
d.define_adj(2,3,weight=compute_weights(position_list[2],position_list[3]))
d.define_adj(3,4,weight=compute_weights(position_list[3],position_list[4]))
d.define_adj(1,4,weight=compute_weights(position_list[1],position_list[4]))
d.define_adj(3,5,weight=compute_weights(position_list[3],position_list[5]))
d.define_adj(6,5,weight=compute_weights(position_list[6],position_list[5]))
d.define_adj(6,7,weight=compute_weights(position_list[6],position_list[7]))
d.define_adj(5,7,weight=compute_weights(position_list[5],position_list[7]))
d.define_adj(4,6,weight=compute_weights(position_list[4],position_list[6]))
d.define_adj(3,6,weight=compute_weights(position_list[3],position_list[6]))
d.define_adj(2,5,weight=compute_weights(position_list[2],position_list[5]))
d.define_adj(8,10,weight=compute_weights(position_list[8],position_list[10]))
d.define_adj(8,9,weight=compute_weights(position_list[8],position_list[9]))
d.define_adj(10,9,weight=compute_weights(position_list[10],position_list[9]))
d.define_adj(6,9,weight=compute_weights(position_list[6],position_list[9]))
d.define_adj(7,9,weight=compute_weights(position_list[7],position_list[9]))
d.define_adj(5,8,weight=compute_weights(position_list[5],position_list[8]))
d.define_adj(5,9,weight=compute_weights(position_list[5],position_list[9]))
d.define_adj(7,8,weight=compute_weights(position_list[7],position_list[8]))
d.define_adj(2,10,weight=compute_weights(position_list[2],position_list[10]))
d.define_adj(5,10,weight=compute_weights(position_list[5],position_list[10]))

for i in nodes_connected_start:
    d.define_adj(0,i,weight=compute_weights(position_list[0],position_list[i]))
for i in nodes_connected_goal:
    d.define_adj(11,i,weight=compute_weights(position_list[11],position_list[i]))  


result_path = d.dijsktra(0,11)

coord_animate = []
for i in result_path:
    coord_animate.append(position_list[i])

lenght = len(result_path) - 1

num_list_add = 11 - len(coord_animate)

for i in range(0,num_list_add):
    coord_animate.append([0,0])




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ax.set_xlim(0,20)
# ax.set_ylim(0,20)

x_data = []
y_data = []

# x1 = point_start[0]
# y1 = point_start[1]
# x2 = position_list[9][0]
# y2 = position_list[9][1]
# x3 = position_list[6][0]
# y3 = position_list[6][1]
# x4 = point_goal[0]
# y4 = point_goal[1]


def points(time):
    if time == lenght:
        return 0
    if time >= 5:
        if coord_animate[6] == [0,0]:
            return np.array([(1-(time-5))*coord_animate[2][0] +(time-5)*coord_animate[3][0], 
                         (1-(time-5))*coord_animate[2][1] +(time-5)*coord_animate[3][1]])
        else:
            return np.array([(1-(time-5))*coord_animate[5][0] +(time-5)*coord_animate[6][0], 
                         (1-(time-5))*coord_animate[5][1] +(time-5)*coord_animate[6][1]]) 
        
    elif time >= 4:
        if coord_animate[5] == [0,0]:
            return np.array([(1-(time-4))*coord_animate[2][0] +(time-4)*coord_animate[3][0], 
                         (1-(time-4))*coord_animate[2][1] +(time-4)*coord_animate[3][1]])
        else:
            return np.array([(1-(time-4))*coord_animate[4][0] +(time-4)*coord_animate[5][0], 
                         (1-(time-4))*coord_animate[4][1] +(time-4)*coord_animate[5][1]])  
    
    elif time >= 3:
        if coord_animate[4] == [0,0]:
            return np.array([(1-(time-3))*coord_animate[2][0] +(time-3)*coord_animate[3][0], 
                         (1-(time-3))*coord_animate[2][1] +(time-3)*coord_animate[3][1]])
        else:
            return np.array([(1-(time-3))*coord_animate[3][0] +(time-3)*coord_animate[4][0], 
                         (1-(time-3))*coord_animate[3][1] +(time-3)*coord_animate[4][1]])    
    elif time >= 2:
        return np.array([(1-(time-2))*coord_animate[2][0] +(time-2)*coord_animate[3][0], 
                         (1-(time-2))*coord_animate[2][1] +(time-2)*coord_animate[3][1]])
    elif time >= 1:
        return np.array([(1-(time-1))*coord_animate[1][0] +(time-1)*coord_animate[2][0], 
                         (1-(time-1))*coord_animate[1][1] +(time-1)*coord_animate[2][1]])
    else:
        return np.array([(1-time)*coord_animate[0][0] +time*coord_animate[1][0], 
                         (1-time)*coord_animate[0][1] +time*coord_animate[1][1]])
    

point, = ax.plot(point_start[0],point_start[1],marker = 'o', color='red')

def update(time):
    result = points(time)
    x_data.append(result[0])
    y_data.append(result[1])
    point.set_xdata(x_data)
    point.set_ydata(y_data)

    # point.set_xdata(result[0])
    # point.set_ydata(result[1])

    return point,

ani = FuncAnimation(fig, update, interval=5, blit=True, repeat=False, frames=np.arange(0,lenght,0.01))
plt.savefig('visibility_graph.png')
plt.show()