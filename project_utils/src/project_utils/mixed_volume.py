"""
Implementation of mixed_volume_bound on the number of realizations. If you want to experiment with it you 
will need to install PHCpy 1.1.4. On Linux you can install `python3-phcpy` with `apt`. This installs the system-wide
version with pre-built optimized binary library.
"""
import numpy as np
import sympy as sp
import networkx as nx

from phcpy import solver, volumes
from project_utils.graph import convert_int_to_networkx

def _random_realization(G):
    """
    Return a random realization of the graph.
    """
    return {v: np.random.rand(2) * 10 for v in G.nodes}


def _realization_to_edge_lengths(G, realization):
    for u, v in G.edges:
        diff = realization[u] - realization[v]
        G.edges[u, v]['len'] = np.linalg.norm(diff)


def _system_of_equations(G, fixed_edge):
    x, y, s = {}, {}, {}
    u, v = fixed_edge

    # Create symbols
    for w in G.nodes:
        x[w] = sp.Symbol(f'x{w}')
        y[w] = sp.Symbol(f'y{w}')
        s[w] = sp.Symbol(f's{w}')

    # Fix two vertices: u at (0,0), v at (L,0)
    L = G.edges[u, v]['len']
    x[u], y[u], s[u] = sp.Integer(0), sp.Integer(0), sp.Integer(0)
    x[v], y[v], s[v] = sp.Float(L), sp.Integer(0), sp.Float(L)**2
    for w in G.neighbors(u):
        s[w] = sp.Float(G.edges[u, w]['len'])**2

    # Fix one triangle vertex if available
    triangle = set(G.neighbors(u)) & set(G.neighbors(v))
    triangle_vertex = None
    for w in triangle:
        L_uw = sp.Float(G.edges[u, w]['len'])
        L_vw = sp.Float(G.edges[v, w]['len'])
        x[w] = (x[v]**2 + L_uw**2 - L_vw**2) / (2 * x[v])
        y[w] = sp.sqrt(L_uw**2 - x[w]**2)
        triangle_vertex = w
        break

    # Edge equations
    edge_eqs = []
    for u_, v_ in G.edges:
        L2 = sp.Float(G.edges[u_, v_]['len']) ** 2
        eq = s[u_] + s[v_] - 2 * x[u_] * x[v_] - 2 * y[u_] * y[v_] - L2
        edge_eqs.append(eq)

    # Sphere equations
    sphere_eqs = [s[w] - (x[w]**2 + y[w]**2) for w in G.nodes]

    eqs = edge_eqs + sphere_eqs

    # Remove 0=0 equations
    eqs = [eq for eq in eqs if not eq.is_Number]
    
    return eqs, triangle_vertex


def _compute_mixed_volume(eqs, triangle_fixed, demics):
    eqs_str = [str(eq) + ';' for eq in eqs]
    if not solver.is_square(eqs_str):
        raise ValueError("Non-square system")
    multiplier = 2 if triangle_fixed is not None else 1
    # demics=False is slower, but correct
    return volumes.mixed_volume(eqs_str, demics=demics) * multiplier


def mixed_volume(graph: nx.Graph, fixed_edge=None, full_list=False, demics=False) -> int | list:
    """
    Cannot be used in multithreaded contexts.
    """

    G = graph.copy()
    
    realization = _random_realization(G)
    _realization_to_edge_lengths(G, realization)

    if fixed_edge:
        eqs, tri_fixed = _system_of_equations(G, fixed_edge)
        return _compute_mixed_volume(eqs, tri_fixed, demics)

    results = []
    eqs_debug = []
    for e in G.edges:
        eqs, tri_fixed = _system_of_equations(G, e)
        eqs_debug.append((e, eqs, tri_fixed))
        mv = _compute_mixed_volume(eqs, tri_fixed, demics)
        results.append((e, mv))

    return results if full_list else min(mv for _, mv in results)


if __name__ == "__main__":
    # Tests

    # 9 vertices Laman graph
    G = 44194926136
    G = convert_int_to_networkx(G, 9)
    
    print("Case where m-Bezout bound and mixed volume differs for fixed edges (5, 6) and (0, 6):",
          mixed_volume(G, full_list=True))
    # LN 224
    # MV [256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 512, 512, 512, 1216, 1248]
    # BB [256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 512, 512, 512, 1280, 1280]

    # L4 graph
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 2), (3, 1)])
    print(f"Min L4 graph mixed volume: {mixed_volume(G, fixed_edge=(0, 1))}")
    assert mixed_volume(G, fixed_edge=(0, 1)) == 4, "Expected mixed volume to be 4"

    # L56 graph
    G = nx.Graph({1: [2, 4, 5], 2: [1, 6, 3], 3: [2, 4, 7], 4: [1, 3, 5, 7], 5: [1, 4, 6], 6: [2, 5, 7], 7: [3, 4, 6]})
    mv = mixed_volume(G, fixed_edge=(3, 7))
    print(f"Min L56 graph mixed volume: {mv}")
    assert mixed_volume(G, fixed_edge=(3, 7)) == 64, "Expected mixed volume to be 64"
    assert mixed_volume(G) == 64, "Expected mixed volume to be 64"


    # Desargues graph
    G = nx.Graph([(0, 1), (0, 2), (0, 4), (1, 2), (1, 5), (2, 3), (3, 4), (3, 5), (4, 5)])
    print(f"Desargues graph mixed volume: {mixed_volume(G)}")
    assert mixed_volume(G) == 32, "Expected Desargues graph mixed volume to be 32"

    # Another L56 variant
    G = nx.Graph([(0, 1), (0, 2), (0, 4), (0, 6), (1, 2), (1, 5), (2, 3), 
                (3, 6), (3, 5), (4, 5), (4, 6)])
    print(f"Minimum L56 variant mixed volume: {mixed_volume(G)}")
    assert mixed_volume(G) == 64, "Expected mixed volume to be 64"

    # L136 graph
    G = nx.Graph([(0, 1), (0, 3), (0, 7), (1, 2), (1, 4), (1, 6), (2, 3), 
                (2, 4), (3, 5), (3, 7), (4, 5), (5, 6), (6, 7)])

    mvs = mixed_volume(G, full_list=True)    
    print(f"Minimum L136 graph mixed volume: {min(mv for _, mv in mvs)}")
    
    print("Testing random realization of L136...")
    for i in range(0, 50):
        mvs = mixed_volume(G, full_list=True)
        
        expected_mvs = [((0, 1), 192), ((0, 3), 192), ((0, 7), 192), ((1, 2), 192), ((1, 4), 192), ((1, 6), 256), 
                        ((3, 2), 192), ((3, 5), 256), ((3, 7), 192), ((7, 6), 256), ((2, 4), 192), ((4, 5), 256), 
                        ((6, 5), 320)]
        if mvs != expected_mvs:
            print('Computed', mvs)
            print('Expected', expected_mvs)
        assert mvs == expected_mvs, "Mixed volumes list mismatch"

    # Jackson-Owen graph
    edges = [(0, 4), (0, 5), (0, 7), (1, 3), (1, 5), (1, 7), (2, 3), 
            (2, 4), (2, 7), (3, 6), (4, 6), (5, 6), (6, 7)]
    G = nx.Graph(edges)
    print(f"Jackson-Owen graph mixed volume: {mixed_volume(G)}")

    # 18-vertex Laman graph (takes long to compute)
    # edges = [(0, 14), (0, 15), (0, 17), (1, 14), (1, 15), (1, 16), (2, 11), 
    #         (2, 12), (2, 17), (3, 10), (3, 13), (3, 16), (4, 7), (4, 8), 
    #         (4, 9), (4, 13), (5, 6), (5, 12), (6, 11), (6, 13), (6, 17), 
    #         (7, 10), (7, 12), (7, 16), (8, 9), (8, 10), (8, 15), (9, 11), 
    #         (9, 14), (10, 17), (11, 16), (12, 14), (13, 15)]
    # G = nx.Graph(edges)
    # print(f"18-vertex Laman graph mBézout bound: {mixed_volume(G)}")


    # This test was created to check if computation of mixed volume was deterministic.
    # It appeared that with demics=True it is not deterministic. So MixedVol be used instead.
    print("Testing correctness of demics, correct value is 15360:")
    from phcpy.dimension import set_seed
    set_seed(1)
    eqs = ['s8 - 2*x1*x8 - 2*y1*y8 - 0.40473494712009;', 's10 - 2*x1*x10 - 2*y1*y10 - 11.4233346356371;', 
           's11 - 2*x1*x11 - 2*y1*y11 - 46.0357773444872;', 's4 - 2.86122585371804*x4 - 15.5433993924025;', 
           's5 - 2.86122585371804*x5 + 0.0909235657124761;', 's3 - 2*x3*x7 - 2*y3*y7 - 25.773905041783;', 
           's4 - 2*x4*x7 - 2*y4*y7 - 7.29080779445411;', 's2 - 2*x2*x9 - 2*y2*y9 - 80.9691895356351;', 
           's3 - 2*x3*x9 - 2*y3*y9 + 10.9112208778073;', 's2 + s8 - 2*x2*x8 - 2*y2*y8 - 68.7471327104426;', 
           's5 + s8 - 2*x5*x8 - 2*y5*y8 - 17.238200546543;', 's10 + s4 - 2*x10*x4 - 2*y10*y4 - 4.72367678597057;', 
           's10 + s5 - 2*x10*x5 - 2*y10*y5 - 3.24007101369439;', 's11 + s2 - 2*x11*x2 - 2*y11*y2 - 55.833858859123;',
             's11 + s3 - 2*x11*x3 - 2*y11*y3 - 133.069127582;', 's2 + s4 - 2*x2*x4 - 2*y2*y4 - 57.5868732021854;', 
             's3 + s5 - 2*x3*x5 - 2*y3*y5 - 26.7779778709238;', '-x1**2 - y1**2 + 3.19147848003255;', 
             '-x7**2 - y7**2 + 4.3553068121243;', '-x9**2 - y9**2 + 28.2156540031585;', 's8 - x8**2 - y8**2;', 
             's10 - x10**2 - y10**2;', 's11 - x11**2 - y11**2;', 's2 - x2**2 - y2**2;', 's4 - x4**2 - y4**2;', 
             's3 - x3**2 - y3**2;', 's5 - x5**2 - y5**2;']
    cnt = 0

    for i in range(10):
        mv = volumes.mixed_volume(eqs, demics=True, vrblvl=0)
        print(mv)
    # Output:
    # 14848
    # 16896
    # 15872
    # 14656
    # 15360
    # 14336
    # 14336
    # 15488
    # 15360
    # 15104
