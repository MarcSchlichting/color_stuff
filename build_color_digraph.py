import networkx
import numpy as np
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import cvxpy
import matplotlib.pyplot as plt
import pickle

n_points = 201
x = np.linspace(0,10,n_points)
y1 = np.exp(-x) + 0.05 * np.random.randn(n_points)
y2 = 2/(1+np.exp(x-3))  + 0.05 * np.random.randn(n_points)

def display_plots(c1,c2,c3,c4):
    plt.ion()
    fig,axs = plt.subplots(1,2,figsize = (8,3))
    axs[0].plot(x,y1,c=c1)
    axs[0].plot(x,y2,c=c2)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("A")
    axs[1].plot(x,y1,c=c3)
    axs[1].plot(x,y2,c=c4)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("B")
    plt.tight_layout()
    plt.show(block=False)
    decision = 1 if input("Enter your response: ")=="A" else 0
    plt.close()

    return decision


g = networkx.DiGraph()
g2 = networkx.DiGraph()

c1 = np.random.rand(3)
c2 = np.random.rand(3)
c3 = np.random.rand(3)

g.add_node(0,colors={"c0":c1,"c1":c2})
g.add_node(1,colors={"c0":c1,"c1":c3})

g2.add_node(0)
g2.add_node(1)

decision = display_plots(c1,c2,c1,c3)


# decision = np.random.randint(2)

if decision == 1:
    #liked first combination better than second
    g.add_edge(0,1)

    g2.add_edge(0,1)
else:
    #liked second combination better
    g.add_edge(1,0)
 
    g2.add_edge(1,0)

for i in range(10):
    print(f"Step {i}")
    #i denotes the node where to expand from
    new_c = np.random.rand(3)
    g.add_node(i+2,colors={"c0":g.nodes[i]["colors"]["c1"],"c1":new_c})

    g2.add_node(i+2)

    decision = display_plots(g.nodes[i]["colors"]["c1"],g.nodes[i]["colors"]["c0"],g.nodes[i]["colors"]["c1"],new_c)

    if decision == 1:
        #liked first combination better
        g.add_edge(i,i+2)

        g2.add_edge(i,i+2)
    else:
        #liked second combination better
        g.add_edge(i+2,i)

        g2.add_edge(i+2,i)

print("Evaluating head and leaf nodes...")

head_nodes = [i for i in range(len(list(g.nodes()))) if g.in_degree(i)==0]
leaf_nodes = [i for i in range(len(list(g.nodes()))) if g.out_degree(i)==0]

current_root = head_nodes[-1]
for hn in reversed(head_nodes[:-1]):
    decision = display_plots(g.nodes[current_root]["colors"]["c0"],
                             g.nodes[current_root]["colors"]["c1"],
                             g.nodes[hn]["colors"]["c0"],
                             g.nodes[hn]["colors"]["c1"])

    if decision == 1:
        g.add_edge(current_root,hn)
        g2.add_edge(current_root,hn)
    
    else:
        #the second one is better aka the new root node
        g.add_edge(hn,current_root)
        g2.add_edge(hn,current_root)

        current_root = hn

current_leaf = leaf_nodes[-1]
for ln in reversed(leaf_nodes[:-1]):
    decision = display_plots(g.nodes[current_leaf]["colors"]["c0"],
                             g.nodes[current_leaf]["colors"]["c1"],
                             g.nodes[ln]["colors"]["c0"],
                             g.nodes[ln]["colors"]["c1"])

    if decision == 1:
        #the first (current leaf) is better aka the new leaf node changes to ln
        g.add_edge(current_leaf,ln)
        g2.add_edge(current_leaf,ln)

        current_leaf = ln
    
    else:
        #the second one is better aka the current leaf node remains the leaf node
        g.add_edge(ln,current_leaf)
        g2.add_edge(ln,current_leaf)


networkx.drawing.nx_pydot.write_dot(g2,"test.dot")
with open("test_graph.pkl","wb") as f:
    pickle.dump(g,f)

x = cvxpy.Variable(22)
objective_terms = []
constraints = []
constraints.append(x[current_leaf]==0)
constraints.append(x[current_root]==1)

for e in g.edges():
    constraints.append(x[e[0]]>=x[e[1]])
    objective_terms.append(cvxpy.square(cvxpy.inv_pos(x[e[0]]-x[e[1]])))

objective = cvxpy.Minimize(cvxpy.sum(objective_terms))
prob = cvxpy.Problem(objective,constraints=constraints)
prob.solve()
print('stop')