"""
Defines S-graphs for the HR algebra a la Courcelle, B. (1993). Graph grammars, monadic second-order logic and the theory of graph minors. In
N. Robertson and P. Seymour (Eds.), Graph Structure Theory, pp. 565—590. AMS.

@author: Meaghan Fowlie
"""

import copy
import logging
import sys
from copy import deepcopy
from typing import Set, Iterable
import penman
from M4LP.HW5.mtool.smatch import get_amr_match, compute_f
from algebra import AlgebraError

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='%(levelname)s (%(name)s) - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class GraphError(Exception):
    def __init__(self, message=None):
        self.message = message


class SGraph:
    """
    s-graphs a la Courcelle et al.
    Nodes can be annotated with extra labels called "sources",
        which can be used to target particular nodes for operations.
    Attributes:
        nodes: set of ints
        edges: dict from nodes to lists of (target, edge label)
        node_labels: dict from node to label
        sources: dict from source to node
        root: int: specially marked node
    """

    def __init__(self, nodes: set[int] = None,
                 edges: dict[int: list[tuple[int, str]]] or None = None,
                 node_labels: dict[int:str] or None = None,
                 sources: dict[str:int] or None = None,
                 root: int or None = None):
        """
        Initialise an SGraph
        Args:
            nodes: a set of ints; if None, default is an empty graph with no nodes.
            edges: a dict from int to lists of (int, str) pairs. The key is the edge origin,
                                                    and the value is the list of edges as (edge target, label) pairs.
            node_labels: dict from int to str: the node labelling function.
            sources: dict from str to int: the source function, mapping sources to nodes.
            root: int: the root node of the graph.
        """
        # check the types of the inputs
        assert nodes is None or isinstance(nodes, Set), f"Nodes must be of type Set but is {type(nodes)}"
        assert edges is None or isinstance(edges, dict), f"Edges must be of type dict but is {type(edges)}"
        assert node_labels is None or isinstance(node_labels,
                                                 dict), f"node_labels must be of type dict but is {type(node_labels)}"
        assert sources is None or isinstance(sources, dict), f"sources must be of type dict but is {type(sources)}"
        assert root is None or isinstance(nodes, Iterable) and root in nodes, f"root must be in nodes"

        # default is an empty graph
        self.nodes = set() if nodes is None else nodes
        self.edges = {} if edges is None else edges
        self.node_labels = {} if node_labels is None else node_labels
        self.sources = {} if sources is None else sources
        self.root = root

        # check the graph makes sense
        assert set(self.edges.keys()).issubset(self.nodes), f"Edges must be subset of nodes x nodes"
        assert all([isinstance(value, list) for value in self.edges.values()]), f"edge values must be lists"
        assert set(self.sources.values()).issubset(self.nodes), f"Sources must be subset of nodes"
        assert set(self.node_labels.keys()).issubset(self.nodes), \
            f"Node labeling refers to non-existent nodes: {set(self.node_labels.keys())} vs {self.nodes}"

    def get_sources_for_node(self, n):
        """
        Finds all sources for a node.
        Args:
            n: int: the node
        Returns: list of strings, the sources.
        """
        sources = []
        for source in self.sources:
            if n == self.sources[source]:
                sources.append(source)
        return sources

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f""
        ret += f"rt:\t\t\t{self.root}\n"
        ret += f"nodes:\t\t{self.nodes}\n"
        ret += f"labels:\t\t{self.node_labels}\n"
        ret += f"sources:\t{self.sources}\n"
        ret += f"edges:\n"
        for n in self.edges:
            for target, label in self.edges[n]:
                ret += f"\t {n} {label} {target}\n"
        return ret

    def __eq__(self, other):
        # uses Smatch to check equality
        if not isinstance(other, SGraph):
            raise NotImplementedError
        else:
            try:
                allowed_error = 0.000001
                g = penman.encode(self.to_penman())
                h = penman.encode(other.to_penman())
                return abs(compute_f(*get_amr_match(g, h))[2] - 1.0) <= allowed_error
            except Exception as e:
                logger.error(f"graphs {self} and {other} can't be compared due to error {e}")
                return False

    def is_root(self, node):
        return node == self.root

    def get_node_label(self, n):
        """
        Returns the label of a node; if unlabelled, returns an empty string.
        Args:
            n: int: the node
        Returns: str
        """
        return self.node_labels.get(n, "")

    def replace_node(self, old_node: int, new_node: int):
        """
        rename a node in place everywhere it appears.
        @param old_node: int.
        @param new_node: int.
        """
        assert new_node not in self.nodes, f"can't replace node {old_node} with {new_node}: it's already present"
        logger.debug(f"replacing {old_node} with {new_node} in \n {self}")
        # root
        if self.root == old_node:
            self.root = new_node
            logger.debug(f"updated root yielding \n{self}")

        # nodes
        self.nodes.remove(old_node)
        self.nodes.add(new_node)
        logger.debug(f"updated nodes to \n{self.nodes}")

        # edges
        if old_node in self.edges:
            # rename the node qua edge source
            edges = self.edges.pop(old_node)
            self.edges[new_node] = edges
        # rename the node qua edge target
        for source in self.edges:
            new_edges = []
            for target, label in self.edges[source]:
                if target == old_node:
                    new_edges.append((new_node, label))
                else:
                    new_edges.append((target, label))
            self.edges[source] = new_edges

        # sources
        for source in self.sources:
            if self.sources[source] == old_node:
                self.sources[source] = new_node

        # node labels
        if old_node in self.node_labels:
            label = self.node_labels.pop(old_node)
            self.node_labels[new_node] = label

    def __add__(self, other):
        """
        Adds two graphs together, keeping the root at the root of self, and merging shared sources.
        This is the Merge function (||) of the HR algebra.
        :param other: SGraph
        :return: SGraph
        """
        assert isinstance(other, type(self))
        # copy both graphs so we don't mess with the originals
        new_self = deepcopy(self)
        new_other = deepcopy(other)

        # rename all nodes in other
        new_node = max(self.nodes.union(other.nodes)) + 1  # avoid all possible conflicts of node names
        for node in other.nodes:
            new_other.replace_node(node, new_node)
            new_node += 1
        # if self and other share any sources, make them the same in other as they are in self.
        for source in self.sources:
            if source in other.sources:
                new_other.replace_node(new_other.sources[source], self.sources[source])

        # update copy of self to include everything in other
        new_self.nodes = self.nodes.union(new_other.nodes)
        for origin in new_other.edges:
            if origin in new_self.edges:
                new_self.edges[origin] += new_other.edges[origin]
            else:
                new_self.edges[origin] = new_other.edges[origin]
        # new_self.edges.update(new_other.edges)
        new_self.sources.update(new_other.sources)
        new_self.node_labels.update(new_other.node_labels)
        return new_self

    def forget(self, source: str):
        """
        remove source (keep node)
        @param source: str
        """
        if source in self.sources:
            self.sources.pop(source)
        else:
            logger.warning(f"No {source}-source to forget")

    def rename(self, old_source: str, new_source: str):
        """
        change source
        @param old_source: str: the source to change
        @param new_source: str: the source to change to
        """
        if new_source in self.sources:
            raise GraphError(f"Can't rename {old_source} to {new_source}: {new_source} already exists")
        if old_source in self.sources:
            node = self.sources.pop(old_source)
            self.sources[new_source] = node

    def add_source(self, node, source):
        """
        add source to node.
        @param source: str: the source to assign to the node.
        @param node: the node to be given the source.
        """
        if source in self.sources:
            if self.sources[source] == node:
                logger.warning(f"{node} already has source {source}")
            else:
                raise GraphError(f"{source} is already present in the graph")
        self.sources[source] = node

    def print_parameters(self):
        """
        Print the graph in such a way that it's easy to build an SGraph by and using copy-paste.
        Returns: str
        """
        ret = f"nodes={self.nodes}, "
        ret += f"edges={self.edges}, "
        ret += f"node_labels={self.node_labels}, "
        ret += f"sources={self.sources}, "
        ret += f"root={self.root}"
        print(ret)
        return ret

    def to_graphviz(self):
        """
        Make a graphviz (dot) representation of the graph.
        :return: str
        """
        ret = "digraph g {\n"
        for node in self.nodes:
            ret += f"{node}"
            label = self.get_node_label(node)
            sources = self.get_sources_for_node(node)
            if label or sources or self.is_root(node):

                ret += " ["
                if self.is_root(node):
                    ret += "style=bold, "
                if label or sources:
                    ret += "label=\""
                    if label:
                        ret += f"{label}"
                    if sources:
                        ret += f"<{','.join(sources)}>"
                ret += "\"]"
            ret += ";\n"

        for node in self.edges:
            for target, label in self.edges[node]:
                ret += f"{node}->{target} [label={label}];\n"
        ret += "}"
        return ret

    def to_penman(self):
        """
        Export to penman.Graph
        Returns: penman.Graph
        """
        triples = []
        nodes_to_add = copy.copy(self.node_labels)

        for source, node in self.sources.items():
            label = f"<{source}>"
            if node in nodes_to_add:
                label = f"{nodes_to_add[node]}{label}"
                nodes_to_add[node] = label
            else:
                triples.append((str(node), ":instance", label))
        for node, label in nodes_to_add.items():
            triples.append((str(node), ":instance", label))

        for source, edges in self.edges.items():
            for target, label in edges:
                triples.append((str(source), label, str(target)))

        g = penman.graph.Graph(triples, top=str(self.root))
        return g
