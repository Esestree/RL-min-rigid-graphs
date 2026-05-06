""""
Implementation of multihomogeneous Bézout bound on the number of embeddings of minimally rigid graphs.
The algorithms and examples implemented here are based on: "On the multihomogeneous Bézout bound on the 
number of embeddings of minimally rigid graphs" https://arxiv.org/abs/2005.14485
"""
import itertools
import networkx as nx

def degrees(n,G):
    """
    Args:
        n (int): Number of vertices.
        G (list of list): List of edges in the form [[i, j], ...].

    Returns:
        list: Vertex degree list for the graph G.
    """
    res = [0] * n
    for edge in G:
        res[edge[0]] += 1 
        res[edge[1]] += 1
    return res


def graph2allfixed(n,G,d):
    """
    Args:
        n (int): Number of vertices.
        G (list of list): List of edges.
        d (int): Dimension.
        sort_degs (int, optional): Unused; for compatibility.

    Returns:
        list: All d-simplices (fully connected subgraphs of d vertices) in G.
    """

    v=[i for i in range(n)]
    
    C = itertools.combinations(v,d)
    fixed=[]
    
    for k in C:
        temp=list(itertools.combinations(k,2))
        flag=1
        temp1=[]
        
        for l in temp:
            temp1.append(l)
            if not l in G:
                flag=0
                break
        
        if flag==1:
            fixed.append(temp1)
            
    return fixed        


def removeEdge(G,indegs,degs,e,dir):
    """
    Creates a copy of the graph G, removes the specified edge from it, and updates the 
    degrees of the affected nodes accordingly.

    Args:
        G (list): List of edges.
        indegs (list): Desired in-degrees before edge removal.
        degs (list): Vertex degrees before edge removal.
        e (int): Index of edge to be removed.
        dir (int): Direction (0 or 1) indicating which vertex to decrement indegree.

    Returns:
        tuple: (Gnew, indegsnew, degsnew) — updated graph and degree lists after removal.
    """
    Gnew = G[:e]+G[e+1:]
    edge = G[e]
    degsnew = degs.copy()
    degsnew[edge[0]] -= 1
    degsnew[edge[1]] -= 1
    indegsnew = indegs.copy()
    indegsnew[edge[dir]] -= 1
    return Gnew,indegsnew,degsnew


def orient(n,G,indegs,degs):
    """
    Args:
        n (int): Number of vertices.
        G (list): List of edges.
        indegs (list): Desired in-degree for each vertex.
        degs (list): Degree of each vertex in G.

    Returns:
        int: Number of different orientations of G with the desired in-degree profile.
    """
    
    idx = 0
    while idx != len(G):
        v, u = G[idx]
        if indegs[v] == 0 or indegs[u] == 0:
            dir = u if indegs[v] == 0 else v
            G[idx] = G[-1]
            G.pop()
            degs[v] -= 1
            degs[u] -= 1
            indegs[dir] -= 1
            idx -= 1
        idx += 1

    idx = 0
    while idx != len(G):
        v, u = G[idx]
        if indegs[v] == degs[v] or indegs[u] == degs[u]:
            dir = v if indegs[v] == degs[v] else u
            G[idx] = G[-1]
            G.pop()
            degs[v] -= 1
            degs[u] -= 1
            indegs[dir] -= 1
            idx -= 1
        idx += 1

    
    for i in range(n):
        if indegs[i] < 0 or indegs[i] > degs[i]:
            return 0
    if len(G) == 0:
        return 1

    Gnew1,indegsnew1,degsnew1 = removeEdge(G,indegs,degs,0,0)
    summand1 = orient(n,Gnew1,indegsnew1,degsnew1)

    Gnew2,indegsnew2,degsnew2 = removeEdge(G,indegs,degs,0,1)
    summand2 = orient(n,Gnew2,indegsnew2,degsnew2)

    return summand1+summand2


def graph2mBezout(n,G,d,fixed):
    """
    Args:
        n (int): Number of vertices.
        G (list): List of edges of a minimally rigid graph.
        d (int): Dimension of the embedding.
        fixed (list or int): A fixed K_d in the form [[i, j], ...], or 1 to compute one automatically.

    Returns:
        int: m-Bézout bound of the system of sphere equations for the given graph in R^d.
    """
    
    indegs = [d] * n
    for edge in fixed:
        G.remove(edge)
        indegs[edge[0]]=0
        indegs[edge[1]]=0
    
    degs=degrees(n,G)
    
    return((2**(n-d))*orient(n,G,indegs,degs))


def mBezout_bound(graph: nx.Graph, fixed_edges=None, full_list=False, dimension=2) -> int | list:
    """
    Args:
        graph (int or nx.Graph): An integer representation of the graph or a NetworkX graph.
        fixed_edges (list, optional): Fixed K_d to use.
        full_list (bool, optional): Whether to return all values or just the minimum.
        dimension (int): Embedding dimension d.

    Returns:
        int or list: Minimum or full list of m-Bézout bounds depending on full_list.
    """
    
    G = graph.copy()
    
    n = G.number_of_nodes()
    G = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    d = dimension

    if fixed_edges:
       fixed_edges = sorted((min(u, v), max(u, v)) for u, v in fixed_edges)
       return graph2mBezout(n, G, d, fixed_edges)

    fixed=graph2allfixed(n,G,d)
    degs_seq=degrees(n,G)
    fixed_degs=[]
    
    for f_i in range(len(fixed)):
        f_temp=fixed[f_i]
        f_s=[]
        for f_j in range(len(f_temp)):
            f_s.extend(f_temp[f_j])
        f_s=list(set(f_s))
        deg_temp=0
        
        for f_k in range(len(f_s)):
            deg_temp=deg_temp+degs_seq[f_s[f_k]]
        fixed_degs.append(deg_temp)
    
    mBezout=[]

    for j in range(len(fixed)):
        G_start=G.copy()
        indegs = [d] * n
        fixed_new=fixed[j]
        
        for edge in fixed_new:
            G_start.remove(edge)
            indegs[edge[0]]=0
            indegs[edge[1]]=0

        degs=degrees(n,G_start)
        orient_cnt = orient(n,G_start,indegs,degs)
        temp= (2**(n-d))*orient_cnt
        mBezout.append(temp) 
    if full_list:
       return mBezout
    else:
        return min(mBezout)


if __name__ == "__main__":
    from project_utils.graph import convert_int_to_networkx
    # Tests

    # Speed test
    G = 71557306413135081116091456327584755953673370689478276983012212630869839968
    G = convert_int_to_networkx(G, 23)
    # print(sorted([min(u, v), max(u, v)] for u, v in G.edges))

    from time import time
    print("Testing speed on graph with 23 vertices: ")
    start = time()
    res = mBezout_bound(G, full_list=True)
    assert res == [2128609280, 2183135232, 1994391552, 2279604224, 2237661184, 2231369728, 2191523840, 
                    2384461824, 2499805184, 2300575744, 2315255808, 2541748224, 2436890624, 2631925760, 
                    2493513728, 2279604224, 2558525440, 2420113408, 1677721600, 1535115264, 1688207360, 
                    1744830464, 1704984576, 1589641216, 1738539008, 1891631104, 1872756736, 1805647872, 
                    1889533952, 2063597568, 1956642816, 1904214016, 2042626048, 1912602624, 1889533952, 
                    1910505472, 1975517184, 1776287744, 2264924160, 2336227328, 2160066560, 2027945984, 
                    2220883968]
    
    print(time() - start) # initial 8.9 s

    # 9 vertices Laman graph
    G = 44194926136
    G = convert_int_to_networkx(G, 9)
    print("Case where m-Bezout bound and mixed volume differs for a fixed edges (5, 6) and (0, 6):\n"
          f"Graph = {sorted([min(u, v), max(u, v)] for u, v in G.edges)}\n"
          f"m-Bezout bounds for each fixed K2: {mBezout_bound(G, full_list=True)}")
    # LN 224
    # MV [256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 512, 512, 512, 1216, 1248]
    # BB [256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 512, 512, 512, 1280, 1280]


    # Laman graphs
    G = nx.Graph([(0, 1), (0, 6), (0, 7), (0, 9), (1, 8), (1, 10), (1, 11),
                  (6, 4), (6, 5), (7, 3), (7, 4), (9, 2), (9, 3), (8, 2),
                  (8, 5), (10, 4), (10, 5), (11, 2), (11, 3), (2, 4), (3, 5)])
    res = mBezout_bound(G, full_list=True, dimension=2)
    print(f"12 vertices bound = {res}")
    assert res == [16384, 15360, 15360, 15360, 15360, 15360, 15360, 16384, 15360, 15360, 15360, 
                   16384, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360]

    G = nx.Graph([[0,1],[0,2],[0,4],[1,2],[1,5],[2,3],[3,4],[3,5],[4,5]])
    res = mBezout_bound(G, dimension=2)
    print(f"Desargues bound = {res}")
    assert res == 32

    G = nx.Graph([[0,1],[0,2],[0,4],[0,6],[1,2],[1,5],[2,3],[3,6],[3,5],[4,5],[4,6]])
    res = mBezout_bound(G, full_list=True, dimension=2)
    print(f"L56 mBézout bounds = {res}")
    assert res == [64, 64, 64, 64, 64, 96, 96, 128, 96, 96, 64]

    G = nx.Graph([[0,1], [0,3], [0,7], [1,2], [1,4], [1,6], [2,3], [2,4], [3,5],
                  [3,7], [4,5], [5,6], [6,7]])
    res = mBezout_bound(G, full_list=True, dimension=2)
    print(f"L136 mBézout bounds = {res}")
    assert res == [192, 192, 192, 192, 192, 256, 192, 192, 256, 192, 256, 320, 256]

    G = nx.Graph([[0,4], [0,5], [0,7], [1,3], [1,5], [1,7], [2,3], [2,4], [2,7],
                  [3,6], [4,6], [5,6], [6,7]])
    res = mBezout_bound(G, full_list=True, dimension=2)
    print(f"J.O. graph bound = {res}")
    assert res == [256, 256, 192, 256, 256, 192, 256, 256, 192, 192, 192, 192, 128]

    G = nx.Graph([[0, 14], [0, 15], [0, 17], [1, 14], [1, 15], [1, 16], [2, 11],
                  [2, 12], [2, 17], [3, 10], [3, 13], [3, 16], [4, 7], [4, 8],
                  [4, 9], [4, 13], [5, 6], [5, 12], [6, 11], [6, 13], [6, 17],
                  [7, 10], [7, 12], [7, 16], [8, 9], [8, 10], [8, 15], [9, 11],
                  [9, 14], [10, 17], [11, 16], [12, 14], [13, 15]])
    res = mBezout_bound(G, dimension=2)
    print(f"18-vertex Laman graph mBézout bound = {res}")
    assert res == 2228224


    #Geiringer graphs
    G = nx.Graph([[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,5],[1,6],[2,3],[2,6],
                  [3,4],[3,6],[4,5],[4,6],[5,6]])
    res = mBezout_bound(G, fixed_edges=[[0,1],[0,2],[1,2]], dimension=3)
    print(f"G48 mBézout bound = {res}")
    assert res == 48

    G = nx.Graph([[0, 1], [0, 5], [0, 7], [0, 8], [0, 11], [1, 2], [1, 5],
                  [1, 6], [1, 8], [2, 3], [2, 6], [2, 8], [2, 9], [3, 4],
                  [3, 6], [3, 9], [3, 10], [4, 5], [4, 6], [4, 10], [4, 11],
                  [5, 6], [5, 11], [7, 8], [7, 9], [7, 10], [7, 11], [8, 9],
                  [9, 10], [10, 11]])
    res = mBezout_bound(G, full_list=True, dimension=3)
    print(f"Icosahedron bound = {res}")
    assert all(r == res[0] for r in res) and res[0] == 54272

