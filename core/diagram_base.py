from graphviz import Digraph

def create_graph(title: str):
    dot = Digraph(comment=title)
    dot.attr(rankdir="LR", fontsize="12")
    return dot
