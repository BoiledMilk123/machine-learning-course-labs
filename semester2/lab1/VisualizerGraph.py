# lab7/VisualizerGraph.py
import networkx as nx
import matplotlib.pyplot as plt

from lab1 import *
from semester2.lab1.init import *


class VisualizerGraph:
    @staticmethod
    def visualize_oil_costs():
        G = nx.Graph()

        for factory_index in range(NUM_FACTORIES):
            factory_node = f'Ф{factory_index + 1} ({PRODUCTION_CREATION[factory_index]})'
            G.add_node(factory_node, color='lightgray')

            for city_index in range(TOTAL_LOCATIONS):
                city_node = f'Г{city_index + 1} ({CITY_DEMAND[city_index]})'
                G.add_node(city_node, color='lightblue')
                G.add_edge(factory_node, city_node, weight=round(OIL_COSTS[factory_index][city_index], 2))

        pos = nx.spring_layout(G, k=0.1, seed=SEED)
        edges = G.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]

        plt.figure(figsize=(12, 8))
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]
        norm = plt.Normalize(min(weights), max(weights))
        edge_colors = plt.cm.RdYlGn_r(norm(weights))

        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=7, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w['weight'] for u, v, w in edges}, font_size=8)
        plt.title("Полный граф стоимости перевозки (за единицу)")
        plt.show()

    @staticmethod
    def visualize_best_solution(best_individual):
        G = nx.Graph()

        for city_index, factory_index in enumerate(best_individual):
            factory_node = f'Ф{factory_index + 1} ({PRODUCTION_CREATION[factory_index]})'
            city_node = f'Г{city_index + 1} ({CITY_DEMAND[city_index]})'
            G.add_node(factory_node, color='lightgray')
            G.add_node(city_node, color='lightblue')
            G.add_edge(factory_node, city_node, weight=round(OIL_COSTS[factory_index][city_index], 2))

        pos = nx.spring_layout(G, k=1.0, seed=SEED)
        edges = G.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]

        plt.figure(figsize=(12, 8))
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]
        norm = plt.Normalize(min(weights), max(weights))
        edge_colors = plt.cm.RdYlGn_r(norm(weights))

        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w['weight'] for u, v, w in edges}, font_size=9)
        plt.title("Оптимальный граф выбора производства для городов")
        plt.show()
