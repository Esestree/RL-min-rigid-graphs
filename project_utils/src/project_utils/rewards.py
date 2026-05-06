"""Module for evaluation of graphs with rewards."""

class NumPlane:
    """Computes the number of plane realizations"""

    def __init__(self):
        """Initializes the NumPlane evaluator with multithreading enabled."""
        self.multithreaded = True

    def __call__(self, graph: int, *args, **kwargs) -> float:
        """Calculates the number of plane realizations for the given minimally rigid graph.

        Args:
            graph (int): Integer representation of the graph.

        Returns:
            float: The computed number of plane realizations .
        """
        from lnumber import lnumber
        Ln = lnumber(graph)
        return Ln


class RealizationsUpperBound:
    """Computes the m-Bezout bound for graph realizations."""

    def __init__(self):
        """Initializes the evaluator with multithreading disabled."""
        self.multithreaded = False

    def __call__(self, graph: int, *args, **kwargs) -> float:
        """Calculates the m-Bezout bound for a networkx-converted graph.

        Args:
            graph (int): Integer representation of the graph.

        Returns:
            float: The m-Bezout upper bound.
        """
        from project_utils.graph import convert_int_to_networkx
        from project_utils.mBezout_bound import mBezout_bound

        n_vertices = (bin(graph).count('1') + 3) // 2
        G = convert_int_to_networkx(graph, n_vertices)
        mBB = mBezout_bound(G)
        return mBB


class NumSphere:
    """Calculates the number of spherical realizations for the given minimally rigid graph."""

    def __init__(self):
        """Initializes the NumSpherical evaluator with multithreading enabled."""
        self.multithreaded = True

    def __call__(self, graph: int, *args, **kwargs) -> float:
        """Calculates the number of spherical realizations for the given minimally rigid graph.

        Args:
            graph (int): Integer representation of the graph.

        Returns:
            float: The computed number of spherical realizations.
        """
        from lnumber import lnumbers
        Sn = lnumbers(graph)
        return Sn


class NumNAC:
    """Calculates the number of NAC colorings for a graph."""

    def __init__(self):
        """Initializes the NumNAC evaluator with multithreading disabled."""
        self.multithreaded = False

    def __call__(self, graph: int, *args, **kwargs) -> float:
        """Computes half the number of NAC colorings.

        Args:
            graph (int): Integer representation of the graph.

        Returns:
            float: The count of unique NAC colorings.
        """
        import pyrigi

        G = pyrigi.Graph.from_int(graph)
        colorings = G.NAC_colorings()
        n_nac_colorings = sum(1 for _ in colorings)
        return n_nac_colorings // 2


class ZeroReward:
    """Returns a constant zero reward."""

    def __init__(self):
        """Initializes the ZeroReward evaluator with multithreading enabled."""
        self.multithreaded = True

    def __call__(self, *args, **kwargs) -> float:
        """Returns zero regardless of input.

        Args:
            *args: Arbitrary positional arguments.

        Returns:
            float: Always returns 0.0.
        """
        return 0
