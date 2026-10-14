import streamlit.components.v1 as components

from repurpose.visualization.graph_builder import graph_html


def show_graph(service, disease):
    components.html(graph_html(service.entities, service.triples, disease), height=480, scrolling=True)
