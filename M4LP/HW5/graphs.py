"""
Defines S-graphs for the HR algebra a la Courcelle, B. (1993). Graph grammars, monadic second-order logic and the theory
 of graph minors. In N. Robertson and P. Seymour (Eds.), Graph Structure Theory, pp. 565—590. AMS.

@author: Meaghan Fowlie
"""

import copy
import logging
import sys
from copy import deepcopy
from typing import Set, Iterable, List, Dict, Tuple
import penman

from .algebra import AlgebraError
from .mtool.smatch import get_amr_match, compute_f

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='%(levelname)s (%(name)s) - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class GraphError(Exception):
    def __init__(self, message=None):
        self.message = message


class Graph:
    """
    Attributes:
        nodes: set of ints
        edges: dict from nodes to lists of (target, edge label) or target if unlabeled
        node_labels: dict from node to label
        root: int: specially marked node
    """

    def __init__(self, nodes: Set[int] = None,
                 edges: Dict[int: List[Tuple[int, str] or int]] or None = None,
                 node_labels: Dict[int:str] or None = None,
                 root: int or None = None):
        """
        Initialise a Graph
        Args:
            nodes: a set of ints; if None, default is an empty graph with no nodes.
            edges: a dict from int to lists of (int, str) pairs. The key is the edge origin,
                                                    and the value is the list of edges as (edge target, label) pairs
                                                    or just edge target if unlabeled.
            node_labels: dict from int to str: the node labelling function.
            root: int: the root node of the graph, default None.
        """
        # check the types of the inputs
        assert nodes is None or isinstance(nodes, Set), f"Nodes must be of type Set but is {type(nodes)}"
        assert edges is None or isinstance(edges, dict), f"Edges must be of type dict but is {type(edges)}"
        assert node_labels is None or isinstance(node_labels,
                                                 dict), f"node_labels must be of type dict but is {type(node_labels)}"
        assert root is None or isinstance(nodes, Iterable) and root in nodes, f"root must be in nodes"

        # default is an empty graph
        self.nodes = set() if nodes is None else nodes
        self.edges = {} if edges is None else edges
        self.node_labels = {} if node_labels is None else node_labels
        self.root = root

        # check the graph makes sense
        assert set(self.edges.keys()).issubset(self.nodes), f"Edges must be subset of nodes x nodes"
        assert all([isinstance(value, list) for value in self.edges.values()]), f"edge values must be lists"
        assert set(self.node_labels.keys()).issubset(self.nodes), \
            f"Node labeling refers to non-existent nodes: {set(self.node_labels.keys())} vs {self.nodes}"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = ""
        if self.root is not None:
            ret += f"rt:\t\t{self.root}\n"
        ret += f"nodes:\t\t{self.nodes}\n"
        if self.node_labels:
            ret += f"labels:\t\t{self.node_labels}\n"
        if self.edges:
            ret += f"edges:\n"
            for n in self.edges:
                for edge in self.edges[n]:
                    if isinstance(edge, tuple):
                        target, label = edge
                        ret += f"\t {n} -{label}-> {target}\n"
                    else:
                        ret += f"\t {n} -> {edge}\n"
        return ret
        
    def __eq__(self, other):
        # uses Smatch to check equality
        if not isinstance(other, type(self)):
            raise NotImplementedError(f"Compared object is not a Graph, but rather a {type(other)}")
        else:
            try:
                allowed_error = 0.000001
                g = penman.encode(self.to_penman())
                h = penman.encode(other.to_penman())
                return abs(compute_f(*get_amr_match(g, h))[2] - 1.0) <= allowed_error
            except Exception as e:
                logger.error(f"graphs {self} and {other} can't be compared due to error {e}")
                return False
                
    def add_edge(self, origin, edge):
        if origin not in self.edges:
            self.edges[origin] = [edge]
        else:
            if isinstance(edge, tuple):
                target, label = edge
            else:
                target, label = edge, None
            targets = self.get_targets(origin)
            if target not in targets:
                self.edges[origin].append(edge)
            else:
                if label is not None:
                    # we'll add a label to this unlabeled node from the other graph
                    if not isinstance(self.edges[origin], tuple):
                        self.edges[origin].remove(target)
                    # let's try allowing multiple edges between nodes
                    self.edges[origin].append(edge)
                
    def get_targets(self, origin):
        """
        Given a node `origin`, find all nodes `target` such that (origin, target) is an edges
        """
        return [edge if not isinstance(edge, tuple) else edge[0] for edge in self.edges[origin]]
                
    def remove_node(self, node):
        self.nodes.remove(node)
        if node in self.node_labels:
            del self.node_labels[node]
        if node in self.edges:
            del self.edges[node]
        for origin in self.edges:
            targets = self.get_targets(origin)
            new_edges = [edge for target, edge in zip(targets, self.edges[origin]) if target != node]
            self.edges[origin] = new_edges
        if self.root == node:
            self.root = None

    def __add__(self, other):
        """
        Adds two graphs together, keeping the root at the root of self, 
        and otherwise simply taking the union of the nodes, edges, and labels.
        :param other: SGraph
        :return: SGraph
        """
        assert isinstance(other, type(self))
        new_labels = deepcopy(self.node_labels)
        for node in other.node_labels:
            if node not in new_labels:
                new_labels[node] = deepcopy(other.node_labels[node])
            elif new_labels[node] != other.node_labels[node]:
                raise GraphError(f"nodes can only have one label, but the two graphs have different labels for {node}")
        
        new_graph = type(self)(self.nodes | other.nodes, deepcopy(self.edges), new_labels, self.root)
        for origin in other.edges:
            new_graph.add_edge(origin, deepcopy(other.edges[origin]))
        return new_graph
        
        

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



    def print_parameters(self):
        """
        Print the graph in such a way that it's easy to build an SGraph by and using copy-paste.
        Returns: str
        """
        ret = f"nodes={self.nodes}, "
        ret += f"edges={self.edges}, "
        ret += f"node_labels={self.node_labels}, "
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
            if label or self.is_root(node):
                ret += " ["
                if self.is_root(node):
                    ret += "style=bold, "
                if label:
                    ret += "label=\""
                    if label:
                        ret += f"{label}"
                ret += "\"]"
            ret += ";\n"

        for node in self.edges:
            for edge in self.edges[node]:
                if isinstance(edge, tuple):
                    target, label = edge
                    ret += f"{node}->{target} [label={label}];\n"
                else:
                    ret += f"{node}->{edge};\n"
                
        ret += "}"
        return ret

    def to_penman(self):
        """
        Export to penman.Graph
        Returns: penman.Graph
        """
        triples = []
        nodes_to_add = copy.copy(self.node_labels)

        for node, label in nodes_to_add.items():
            triples.append((str(node), ":instance", label))

        for source, edges in self.edges.items():
            for edge in edges:
                if isinstance(edge, tuple):
                    target, label = edge
                else:
                    target, label = edge, ""
                triples.append((str(source), label, str(target)))

        g = penman.graph.Graph(triples, top=str(self.root))
        return g


class SGraph(Graph):
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
                 edges: dict[int: list[tuple[int, str] or int]] or None = None,
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
        super().__init__(nodes, edges, node_labels, root)
        self.sources = {} if sources is None else sources


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

    def __repr__(self):
        ret = super().__repr__()
        ret += f"sources:\t\t{self.sources}\n"
        return ret

    def _replace_node(self, old_node: int, new_node: int):
        """
        Private function, not meant for use outside of self.__add__.
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
            for edge in self.edges[source]:
                if isinstance(edge, tuple):
                    target, label = edge
                else:
                    target = edge
                    label = None
                if target == old_node:
                    if label is not None:
                        new_edges.append((new_node, label))
                    else: 
                        new_edges.append(new_node)
                else:
                    new_edges.append(edge)
            self.edges[source] = new_edges

        # node labels
        if old_node in self.node_labels:
            label = self.node_labels.pop(old_node)
            self.node_labels[new_node] = label
            
        # sources
        for source in self.sources:
            if self.sources[source] == old_node:
                self.sources[source] = new_node

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
            new_other._replace_node(node, new_node)
            new_node += 1
        # if self and other share any sources, make them the same in other as they are in self.
        for source in self.sources:
            if source in other.sources:
                new_other._replace_node(new_other.sources[source], self.sources[source])

        # update copy of self to include everything in other
        new_self.nodes = self.nodes.union(new_other.nodes)
        for origin in new_other.edges:
            if origin in new_self.edges:
                new_self.edges[origin] += new_other.edges[origin]
            else:
                new_self.edges[origin] = new_other.edges[origin]

        for source in new_other.sources:
            if source in new_self.sources and new_self.sources[source] in new_self.node_labels and new_other.sources[source] in new_other.node_labels:
                raise AlgebraError(f"source {source} already has a label")
        new_self.sources.update(new_other.sources)
        new_self.node_labels.update(new_other.node_labels)
        return new_self

    def forget(self, source: str):
        """
        HR algebra operation: remove source (keep node)
        @param source: str
        """
        if source in self.sources:
            self.sources.pop(source)
        else:
            logger.warning(f"No {source}-source to forget")

    def rename(self, old_source: str, new_source: str):
        """
        HR algebra operation: change source
        @param old_source: str: the source to change
        @param new_source: str: the source to change to
        """
        if new_source in self.sources:
            raise GraphError(f"Can't rename {old_source} to {new_source}: {new_source} already exists")
        if old_source in self.sources:
            node = self.sources.pop(old_source)
            self.sources[new_source] = node

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
        
if __name__ == "__main__":
    g = Graph({1,2,3}, edges={1:[2, (3, "hi")]})
    print(g)
    
    g.remove_node(2)
    print(g)
