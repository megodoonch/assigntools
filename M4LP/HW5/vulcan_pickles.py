# import useful things from Vulcan
from vulcan.pickle_builder.pickle_builder import PickleBuilder
from vulcan.data_handling.format_names import FORMAT_NAME_GRAPH, FORMAT_NAME_STRING, FORMAT_NAME_NLTK_TREE
from .graphs import SGraph, Graph

# provided functions for creating vulcan-readable pickles




def create_vulcan_pickle_of_graphs(graphs, pickle_path, comments=None):
    """
    Creates a vulcan-readable pickle of a list of SGraphs.
    If there is an error converting one of your graphs to a penman.Graph,
        a graph with one node labelled with the error is created instead.
    :param graphs: list of SGraphs to visualise
    :param pickle_path: path to write the pickle to (including pickle name)
    :param comments: optional list of strings, one for each graph.
    """

    if comments is None:
        comments = [""]*len(graphs)

    my_pickle_builder = PickleBuilder({"Graph": FORMAT_NAME_GRAPH, "Comments": FORMAT_NAME_STRING})
    for graph, comment in zip(graphs, comments):
        try:
            student_penman = graph.to_penman()
        except Exception as e:
            student_penman = SGraph(nodes={0}, node_labels={0: f"Error converting to Penman: {e}"}, root=0).to_penman()
            print(student_penman)
        my_pickle_builder.add_instances_by_name({
            "Graph": student_penman,
            "Comments": comment
        })
    my_pickle_builder.write(pickle_path)


def create_vulcan_pickle_gold_and_student_graphs(gold_graphs, student_graphs, pickle_path, comments=None):
    """
    Creates a vulcan-readable pickle comparing the correct graphs to your own graphs.
    If there is an error converting one of your graphs to a penman.Graph,
        a graph with one node labelled with the error is created instead.
    :param gold_graphs: list of correct SGraphs
    :param student_graphs: list of students' SGraphs
    :param pickle_path: path to write the pickle to (including pickle name)
    :param comments: optional list of strings, one for each graph pair.
    """
    if comments is None:
        comments = [""]*len(gold_graphs)
    my_pickle_builder = PickleBuilder({"Student AMR": FORMAT_NAME_GRAPH, "Gold AMR": FORMAT_NAME_GRAPH, "Comments": FORMAT_NAME_STRING})
    for (student_graph, gold), comment in zip(zip(student_graphs, gold_graphs), comments):
        try:
            student_penman = student_graph.to_penman()
        except Exception as e:
            student_penman = SGraph(nodes={0}, node_labels={0: f"Error converting to Penman: {e}"}, root=0).to_penman()
            print(student_penman)
        my_pickle_builder.add_instances_by_name({
            "Student AMR": student_penman,
            "Gold AMR": gold.to_penman(),
            "Comments": comment
        })
    my_pickle_builder.write(pickle_path)



def create_vulcan_pickle_terms_and_graphs(terms: list[AlgebraTerm], pickle_path, comments=None):
    """
    Creates a vulcan-readable pickle of a list of algebra terms and SGraphs.
    If there is an error converting one of your graphs to a penman.Graph,
        a graph with one node labelled with the error is created instead.
    :param graphs: list of AlgebraTerms to visualise and evaluate to graphs
    :param pickle_path: path to write the pickle to (including pickle name)
    :param comments: optional list of strings, one for each graph/term pair.
    """
    if comments is None:
        comments = [""]*len(terms)

    my_pickle_builder = PickleBuilder({"Tree": FORMAT_NAME_NLTK_TREE,
                                       "Graph": FORMAT_NAME_GRAPH,
                                       "Comments": FORMAT_NAME_STRING})
    for term, comment in zip(terms, comments):
        try:
            nltk_tree = term.to_nltk_tree()
        except Exception as e:
            nltk_tree = nltk.Tree(f"Couldn't make NLTK tree from term: {e}")
        try:
            # evaluate the term to get an s-graph
            graph = term.evaluate()
            try:
                student_penman = graph.to_penman()
            except Exception as e:
                student_penman = SGraph(nodes={0}, node_labels={0: f"Error converting to Penman: {e}"}, root=0).to_penman()
        except Exception as e:
            student_penman = SGraph(nodes={0}, node_labels={0: f"Error evaluating term: {e}"}, root=0).to_penman()

        my_pickle_builder.add_instances_by_name({
            "Tree": nltk_tree,
            "Graph": student_penman,
            "Comments": comment
        })
    my_pickle_builder.write(pickle_path)

def create_vulcan_pickle_terms_and_gold_graphs_and_student_graphs(terms: list[AlgebraTerm], gold_graphs: list[SGraph], pickle_path, comments=None):
    """
    Creates a vulcan-readable pickle comparing the correct graphs to your own graphs,
     including the algebra term that evaluates to the graph.
    If there is an error converting one of your graphs to a penman.Graph,
        a graph with one node labelled with the error is created instead.
    :param terms: list of AlgebraTerms to visualise and evaluate to graphs.
    :param gold_graphs: list of correct SGraphs.
    :param pickle_path: path to write the pickle to (including pickle name).
    :param comments: optional list of strings, one for each term/graph/gold triple.
    """
    if comments is None:
        comments = [""]*len(terms)

    my_pickle_builder = PickleBuilder({"Tree": FORMAT_NAME_NLTK_TREE,
                                       "Graph the term evaluates to": FORMAT_NAME_GRAPH,
                                       "Gold Graph": FORMAT_NAME_GRAPH,
                                       "Comments": FORMAT_NAME_STRING})
    for (term, gold), comment in zip(zip(terms, gold_graphs), comments):
        try:
            nltk_tree = term.to_nltk_tree()
        except Exception as e:
            nltk_tree = nltk.Tree(f"Couldn't make NLTK tree from term: {e}")
        try:
            # evaluate the term to get a graph
            graph = term.evaluate()
            try:
                student_penman = graph.to_penman()
            except Exception as e:
                student_penman = SGraph(nodes={0}, node_labels={0: f"Error converting to Penman: {e}"}, root=0).to_penman()
        except Exception as e:
            student_penman = SGraph(nodes={0}, node_labels={0: f"Error evaluating term: {e}"}, root=0).to_penman()

        my_pickle_builder.add_instances_by_name({
            "Tree": nltk_tree,
            "Graph the term evaluates to": student_penman,
            "Gold Graph": gold.to_penman(),
            "Comments": comment
        })
    my_pickle_builder.write(pickle_path)
