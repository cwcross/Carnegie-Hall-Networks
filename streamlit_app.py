import streamlit as st
# import libraries
from community import community_louvain
from copy import deepcopy
import altair as alt
# from IPython.display import SVG
import glob as glob
import numpy as np
import os
import pandas as pd
import re
import networkx as nx
import plotly.express as px
import requests



from pyvis import network as net
import networkx as nx
from community import community_louvain
from itertools import combinations
from copy import deepcopy

# setup plotting for quarto
alt.renderers.enable('default')
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"

# supress warnings
import warnings
warnings.filterwarnings('ignore')



st.title("Carnegie Hall: Networking the New York Philharmonic")



filepath = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTqBUTiOGYRNrU9r46vA44jlbwR4tcVEcT2C3ermJWR783x-EOWNVhKSLwBxbYWmLb8RkFIuP_7e3dm/pub?gid=0&single=true&output=csv'
df_import = pd.read_csv(filepath)
df_import.reset_index(drop=True, inplace=True)

# Clean the data
# Luckily, there is no tidying to do, because everything is organized well by Carnegie
df = df_import
df['event'] = df['event'].str[-5:]
df = df.drop(columns = ['title'])
column_list = ['year','composer','work','event']
df = df[column_list]

def filter_by_year(df,bottom:int,top:int):
    df_floored = df[df['year'] >= bottom]
    df_final = df_floored[df_floored['year'] <= top]
    return df_final




def create_network(df: pd.DataFrame,name:str,min_threshold:int,edge_thinness:int):
    df0 = df
    feature_to_groupby = 'event'
    column_for_list_of_edges = 'composer'
    grouped_feature_with_edges = df0.groupby(feature_to_groupby)[column_for_list_of_edges].unique().reset_index(name=column_for_list_of_edges)
    all_pairs = []
    for _, row in grouped_feature_with_edges.iterrows():
        pairs = list(combinations(row[column_for_list_of_edges], 2))
        all_pairs.append((row[feature_to_groupby], pairs))

    # Add index of composer birthyears
    filepath = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTqBUTiOGYRNrU9r46vA44jlbwR4tcVEcT2C3ermJWR783x-EOWNVhKSLwBxbYWmLb8RkFIuP_7e3dm/pub?gid=596692414&single=true&output=csv'
    df_import = pd.read_csv(filepath)
    df_import.reset_index(drop=True, inplace=True)
    
    def normalize_import(entry):
        string = str(entry)
        short = string[0:4]
        return short
        
    df_import['Born: '] = df_import['ComposerBirthYear'].apply(normalize_import)
    df_import = df_import.drop(columns=['ComposerBirthYear'])
    df_birth = df_import.set_index('composerName')
    
    # Create a new DataFrame with the results
    edge_pair_name = column_for_list_of_edges + "_Pairs"
    edge_pair_df = pd.DataFrame(all_pairs, columns=[feature_to_groupby, edge_pair_name])
    # adjust for a threshold of genres per piece. should be > 0 
    edge_pair_df_filtered = edge_pair_df[edge_pair_df[edge_pair_name].apply(len) > 0]
    exploded_edge_pairs = edge_pair_df_filtered.explode(edge_pair_name)
    pair_counts = exploded_edge_pairs[edge_pair_name].value_counts()
    pair_counts_df = pd.DataFrame(pair_counts).reset_index()
    df_composer_frequency = df['composer'].value_counts()
    # allow a filter for the number of times a given pair of genres occurs
    # this works with the original series of pair counts, not the df
    
    
    minimum_count_for_pair = 50
    
    pair_counts_filtered = (pair_counts[pair_counts >= min_threshold])
    
    # set graph options:
    graph_height = 800
    graph_width = 1200
    detect_louvain_communities = True
    add_forceAtlas2Based_physics = True
    
    # Create an empty NetworkX graph
    G = nx.Graph()
    
    # Add nodes and assign weights to edges
    for pair, count in pair_counts_filtered.items():
        # Directly unpacking the tuple into node1 and node2
        node1, node2 = pair
        # Adding nodes if they don't exist already
        if node1 not in G.nodes:
            if df_import.isin([node1]).any().any():
                G.add_node(node1, value = df_composer_frequency[node1]/65, title = f'Performances: {df_composer_frequency[node1]} {df_birth.loc[node1]}')
            else:
                G.add_node(node1, value = df_composer_frequency[node1]/65, title = f'Performances: {df_composer_frequency[node1]} Name: {node1}')
        if node2 not in G.nodes:
            if df_import.isin([node2]).any().any():
                G.add_node(node2, value = df_composer_frequency[node2]/65, title = f'Performances: {df_composer_frequency[node2]} {df_birth.loc[node2]}')
            else:
                G.add_node(node2, value = df_composer_frequency[node2]/65, title = f'Performances: {df_composer_frequency[node1]} Name: {node1}')
        # Adding edge with weight
        G.add_edge(node1, node2, weight=(count/edge_thinness), title = f'{node1} and {node2} have performed {count} times together.')

    
    # Adjusting edge thickness based on weights
    for edge in G.edges(data=True):
        edge[2]['width'] = edge[2]['weight']
        
    # Adding Louvain Communities
    
    if detect_louvain_communities == True:
        def add_communities(G):
            G = deepcopy(G)
            partition = community_louvain.best_partition(G)
            nx.set_node_attributes(G, partition, "group")
            return G
            
        G = add_communities(G)
    
    # set display parameters
    network_graph = net.Network(notebook=True,
                       width=graph_height,
                       height=graph_height,
                       bgcolor="black", 
                       font_color="white")
    
    # Set the physics layout of the network
    
    if add_forceAtlas2Based_physics == True:
    
        network_graph.set_options("""
        {
        "physics": {
        "enabled": true,
        "forceAtlas2Based": {
            "springLength": 1
        },
        "solver": "forceAtlas2Based"
        }
        }
        """)
    
    network_graph.from_nx(G)
    return network_graph
    # # return the network
    # network_graph.show(f'{name}_Carnegie_NY_Phil.html')

lower_bound = st.number_input('From (inclusive):',min_value=1892,max_value=2022)
upper_bound = st.number_input('Until (inclusive):',min_value=1892,max_value=2022)
df1 = filter_by_year(df,lower_bound,upper_bound)
min_threshold = st.number_input('Minimum number of connected performances:',min_value=1)
thinness = st.number_input('Line Thinness Multiplier:',min_value=1)
title = st.text_input('Network Title:')
st.write(create_network(df1,'network1',min_threshold,thinness).show(f'{title}.html'))