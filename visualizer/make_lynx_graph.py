# %% Taken from sna_hinter
from typing import Dict

import community
import networkx as nx
from generic_iterative_stemmer.models import StemmedKeyedVectors
from gensim.models import KeyedVectors

from codenames.game.builder import words_to_random_board
from codenames.visualizer import render
from language_data.model_loader import load_language

# model = load_language("english", "google-300")
model = KeyedVectors.load_word2vec_format("language_data/english/google-300.bin", binary=True)
words = [
    # "cloak",
    # "kiss",
    # "flood",
    # "mail",
    # "skates",
    # "paper",
    # "frog",
    # "skyscraper",
    # "moon",
    # "egypt",
    "teacher",
    # "avalanche",
    "newton",
    # "violet",
    # "drill",
    # "fever",
    # "ninja",
    "jupiter",
    "ski",
    # "attic",
    # "beach",
    # "lock",
    # "earth",
    "park",
    # "gymnast",
    "water",
]
# %%
board = words_to_random_board(words=words)
board_size = board.size
vis_graph = nx.Graph()
vis_graph.add_nodes_from(board.all_words)
louvain = nx.Graph(vis_graph)
for i in range(board_size):
    v = board.all_words[i]
    for j in range(i + 1, board_size):
        u = board.all_words[j]
        distance = model.similarity(v, u) + 1
        if distance > 1.12:
            vis_graph.add_edge(v, u, weight=distance)
        louvain_weight = distance ** 25
        louvain.add_edge(v, u, weight=louvain_weight)
#
word_to_group: Dict[str, int] = community.best_partition(louvain)
nx.set_node_attributes(vis_graph, word_to_group, "group")
render(vis_graph)
