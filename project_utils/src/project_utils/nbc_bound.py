"""Module for computing non-broken circuit (nbc) bases of graph matroids."""

from time import time

import sympy
import networkx as nx

from project_utils.graph import convert_int_to_networkx
from matplotlib import pyplot as plt   


class DSU:
    """Disjoint Set Union (Union-Find) with component tracking."""

    def __init__(self, n):
        """Initializes DSU with n singleton components.

        Args:
            n (int): Number of elements.
        """
        self.parent = list(range(n))
        self.components = [frozenset([i]) for i in range(n)]

    def find(self, x):
        """Finds the representative of the set containing x with path compression.

        Args:
            x (int): Element to find.

        Returns:
            int: The root representative of the element.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """Unites components containing a and b.

        Args:
            a (int): First element.
            b (int): Second element.
        """
        self.components[b] = self.components[b] | self.components[a]
        self.components[a] = frozenset()
        self.parent[a] = b

    def get_component(self, root):
        """Returns the set of elements in the component rooted at 'root'.

        Args:
            root (int): Root representative.

        Returns:
            frozenset: Set of elements in the component.
        """
        return self.components[root]

    def copy(self):
        """Creates a deep copy of the DSU state.

        Returns:
            DSU: A new DSU instance with the same parent and component lists.
        """
        new = object.__new__(DSU)
        new.parent = self.parent[:]
        new.components = self.components[:]
        return new


def has_broken_circuit(adj, edges_to_check, edge_order):
    """Checks if adding an edge creates a broken circuit.

    Args:
        adj (list): Adjacency list representation of the graph.
        edges_to_check (list): List of edges to verify for cycles.
        edge_order (dict): Mapping from edge tuples to their lexicographical rank.

    Returns:
        bool: True if a broken circuit is found, False otherwise.
    """
    
    def find_cycle(u, visited, edge_order):
        parent = [-1] * len(visited)
        stack = [(u, -1)]

        while stack:
            node, par = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            parent[node] = par

            for nei in adj[node]:
                if nei == par:
                    continue
                if visited[nei]:
                    min_edge = (node, nei)
                    x = node
                    while x != nei:
                        if parent[x] == -1:
                            print(x, parent, adj)
                        min_edge = min(min_edge, (x, parent[x]), key=lambda edge: edge_order[edge])
                        x = parent[x]
                    return min_edge
                stack.append((nei, node))
    

    for e in edges_to_check:
        u, v = e

        adj[u].append(v)
        adj[v].append(u)
        for l in adj:
            if len(set(l)) != len(l):
                print(adj, e)
                exit(0)

        visited = [False] * len(adj)
        min_edge = find_cycle(u, visited, edge_order)
        adj[u].pop()
        adj[v].pop()

        if min_edge == e or min_edge == e[::-1]:
            return True
    
    return False


def count_nbc_bases(G):
    """Counts the number of non-broken circuit bases of a graph G.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        int: The total number of nbc-bases.
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    result = 0
    edge_to_index = {e: i for i, e in enumerate(edges)} | {e[::-1]: i for i, e in enumerate(edges)}
    total_calls = 0

    def backtrack(n_edges, start_idx, adj, dsu):
        nonlocal total_calls
        total_calls += 1
        if n_edges == n - 1:
            nonlocal result
            result += 1
            return
        
        # end of range is calculated from the fact that tree can be build only if: 
        # len(edges) - i - 1 >= (n - 1) - (n_edges + 1)
        for i in range(start_idx, len(edges) + n_edges - n + 2):
            u, v = edges[i]
            pu = dsu.find(u)
            pv = dsu.find(v)
            if pu != pv:
                adj[u].append(v)
                adj[v].append(u)

                C_u = dsu.get_component(pu)
                C_v = dsu.get_component(pv)
                edges_to_check = []
                for e in edges:
                    if e == (u, v):
                        continue
                    if e[0] in C_u and e[1] in C_v:
                        edges_to_check.append(e)
                    elif e[1] in C_u and e[0] in C_v:
                        edges_to_check.append(e)

                if not has_broken_circuit(adj, edges_to_check, edge_to_index):
                    dsu_copy = dsu.copy()
                    dsu_copy.union(pu, pv)
                    backtrack(n_edges + 1, i + 1, adj, dsu_copy)

                adj[u].pop()
                adj[v].pop()

    adj = [[] for _ in range(n)]
    u, v = edges[0]
    adj[u].append(v)
    adj[v].append(u)
    dsu = DSU(n)
    dsu.union(u, v)
    backtrack(1, 1, adj, dsu)

    return result


def nbc_bound(graph: nx.Graph) -> int:
    """Calculates the nbc-bound for a graph.

    Args:
        graph (nx.Graph): The input NetworkX graph.

    Returns:
        int: The nbc-bound for the graph.
    """
    G = graph.copy()
    
    val = count_nbc_bases(G)
    return val


if __name__ == "__main__":
    # Tests
    from project_utils.henneberg_extensions import generate_random_graphs
    from project_utils.graph import convert_bin_to_int
    # from lnumber import lnumber
    # from min_rig_utils.mixed_volume import mixed_volume
    # from min_rig_utils.mBezout_bound import mBezout_bound

    # The graph for which nbc bound is significanlty better
    # G = 8448416810757410123028394673197057
    # print(lnumber(G)) # output 232328
    # print(nbc_bound(G)) # output 650212
    # print(mBezout_bound(G)) # output 819200

    # And the graph for which it is worse
    # G = 127379915342873608763919675076651008
    # print(lnumber(G)) # output 290822
    # print(nbc_bound(G)) # output 935236
    # print(mBezout_bound(G)) # output 737280

    N = 10
    print(f"Testing correctness on random graphs with {N} vertices")
    graphs, _ = generate_random_graphs(2, N)
    graphs = graphs[:, :N*(N-1)//2]
    for G in map(convert_bin_to_int, graphs):
        G = convert_int_to_networkx(G, N)
        pol = nx.tutte_polynomial(G)
        x, y = sympy.symbols('x  y')
        tutte_val = pol.subs({x: 1, y: 0})
        st_prune_val = nbc_bound(G)
        # print(f"Graph: {G}\n"
        #       f"nbc bound:      {st_prune_val}\n"
        #       f"m-Bezout bound: {mBezout_bound(G)}\n"
        #       f"lnumber {lnumber(G)}")

        assert tutte_val == st_prune_val
    
    # For 16 vertices:
    # Graph: 447848333887640992642176733689156112
    # nbc bound:      372822
    # m-Bezout bound: 98304
    # lnumber 48384
    # Graph: 791841237180884441574652703162895632
    # nbc bound:      298228
    # m-Bezout bound: 16384
    # lnumber 16384
    # Graph: 799801967439990079162313915818116096
    # nbc bound:      256064
    # m-Bezout bound: 32768
    # lnumber 24576


    G = nx.Graph([(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3), (1, 4), (2, 5)])
    res = nbc_bound(G)
    print(f"3-Prism nbc bound = {res}")
 
    # res = nbc_bound(G)
    assert res == 26

    # Speed test
    # from time import time
    # print("Testing speed on graph with 23 vertices: ")
    # G = 10174032589967810518073233932480
    # start = time()
    # res = nbc_bound(G)
    # assert res == 1535115264
    # print(time() - start)

    # 9 vertices Laman graph
    G = 44194926136
    G = convert_int_to_networkx(G, 9)
    res = nbc_bound(G)
    print(f"Graph = {sorted([min(u, v), max(u, v)] for u, v in G.edges)}\n"
          f"nbc bound = {res}")
    assert res == 422


    # # Laman graphs
    G = nx.Graph([(0, 1), (0, 6), (0, 7), (0, 9), (1, 8), (1, 10), (1, 11),
                  (6, 4), (6, 5), (7, 3), (7, 4), (9, 2), (9, 3), (8, 2),
                  (8, 5), (10, 4), (10, 5), (11, 2), (11, 3), (2, 4), (3, 5)])
    res = nbc_bound(G)
    print(f"12 vertices bound = {res}")
    assert res == 12528

    G = nx.Graph([[0,1],[0,2],[0,4],[1,2],[1,5],[2,3],[3,4],[3,5],[4,5]])
    res = nbc_bound(G)
    print(f"Desargues nbc bound = {res}")
    assert res == 26

    G = nx.Graph([[0,1],[0,2],[0,4],[0,6],[1,2],[1,5],[2,3],[3,6],[3,5],[4,5],[4,6]])
    res = nbc_bound(G)
    print(f"L56 nbc bound = {res}")
    assert res == 66

    G = nx.Graph([[0,1], [0,3], [0,7], [1,2], [1,4], [1,6], [2,3], [2,4], [3,5],
                  [3,7], [4,5], [5,6], [6,7]])
    res = nbc_bound(G)
    print(f"L136 nbc bound = {res}")
    assert res == 166

    G = nx.Graph([[0,4], [0,5], [0,7], [1,3], [1,5], [1,7], [2,3], [2,4], [2,7],
                  [3,6], [4,6], [5,6], [6,7]])
    res = nbc_bound(G)
    print(f"J.O. graph nbc bound = {res}")
    assert res == 195

    # G = nx.Graph([[0, 14], [0, 15], [0, 17], [1, 14], [1, 15], [1, 16], [2, 11],
    #               [2, 12], [2, 17], [3, 10], [3, 13], [3, 16], [4, 7], [4, 8],
    #               [4, 9], [4, 13], [5, 6], [5, 12], [6, 11], [6, 13], [6, 17],
    #               [7, 10], [7, 12], [7, 16], [8, 9], [8, 10], [8, 15], [9, 11],
    #               [9, 14], [10, 17], [11, 16], [12, 14], [13, 15]])
    # res = nbc_bound(G, dimension=2)
    # print(f"18-vertex Laman graph mBézout bound = {res}")
    # assert res == ?


