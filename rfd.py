'''
Dependencies:
    * networkx
    * matplotlib

Description:
    This script implements the River Formation Dynamics (RFD) algorithm for finding optimal paths on a graph. 
    It simulates water flow and erosion to reinforce paths from an origin node to a destination node. 
    The algorithm is applied to a 2D grid graph and generates 2 images: graph.png and path_altitude.png.
    These images visualize the optimal path as identified by algorithm.
    By default, random weights are assigned to the edges of the graph on each execution.

Execute:
    python rdf.py
'''

import random
from typing import Any
from typing import Hashable

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx


class RFD:
    def __init__(
        self,
        graph: nx.Graph,
        origin: Hashable,
        destination: Hashable,
        n_iter: int | None = None,
    ) -> None:
        """
        Initialize the River Formation Dynamics (RFD) algorithm.

        Args:
            graph (nx.Graph): The input graph.
            origin (Hashable): The starting node.
            destination (Hashable): The destination node.
        """
        self.graph = graph
        self.origin = origin
        self.destination = destination
        self.n_nodes = len(graph.nodes)

        self.altitude_start = 100
        self.default_weight = 1
        self.climbing_param = 1
        self.climbing_param_increment = 0.2
        self.climbing_param_increment_every = 5
        self.n_drops = 100
        self.erosion_param = 20
        self.n_iterations = int((3 / 2) * self.n_nodes) if n_iter is None else n_iter

        self.droplets = {}
        self.erosions = {}
        self.optimal_altitude_path = None

    def initialise_altitudes(self) -> None:
        """
        Initialize the altitudes of all nodes in the graph.
        Sets self.altitude_start for all nodes except the destination, which is set to 0.
        """
        attrs = {node: {"altitude": self.altitude_start} for node in self.graph.nodes()}
        attrs[self.destination] = {"altitude": 0}

        nx.set_node_attributes(self.graph, attrs)

    def initialise_weights(self) -> None:
        """
        Initialize edge weights (distances) in the graph.
        """
        attrs = {
            edge: {"distance": random.randint(1, 10)} for edge in self.graph.edges()
        }

        nx.set_edge_attributes(self.graph, attrs)

    def gradient(self, node_i: Hashable, node_j: Hashable) -> float:
        """
        Calculate the gradient between two nodes.

        Args:
            node_i (Hashable): The source node.
            node_j (Hashable): The destination node.

        Returns:
            float: The gradient value.
        """
        altitude_i = self.graph.nodes[node_i].get("altitude")
        altitude_j = self.graph.nodes[node_j].get("altitude")
        distance_ij = self.graph.edges[(node_i, node_j)].get("distance")

        return (altitude_j - altitude_i) / distance_ij

    def transition_probabilities(self, node_i: Hashable) -> dict[Any, float]:
        """
        Compute transition probabilities to neighbors of a given node.

        Args:
            node_i (Hashable): The current node.

        Returns:
            dict[Any, float]: Transition probabilities to neighboring nodes - key: node, value: transition probability.
        """
        neighbors = list(self.graph.neighbors(node_i))
        gradients = {
            neighbor: self.gradient(node_i, neighbor) for neighbor in neighbors
        }
        total = sum(map(abs, gradients.values()))

        if total == 0:
            # flat environment
            return {neighbor: 1 / self.climbing_param for neighbor in neighbors}

        return {
            neighbor: (
                abs(gradients[neighbor]) / total
                if gradients[neighbor] < 0
                else 1 / self.climbing_param
            )
            for neighbor in neighbors
        }

    def erosion(self, node_i: Hashable, node_j: Hashable) -> float:
        """
        Calculate the erosion amount based on the gradient between two nodes.

        Args:
            node_i (Hashable): The source node.
            node_j (Hashable): The destination node.

        Returns:
            float: The erosion value.
        """
        gradient = self.gradient(node_i, node_j)

        if gradient > 0:
            return 0

        if gradient == 0:
            return self.erosion_param / ((self.n_nodes - 1) * self.n_drops)

        return (
            abs(self.gradient(node_i, node_j))
            * self.erosion_param
            / ((self.n_nodes - 1) * self.n_drops)
        )

    def place_droplets(self) -> None:
        """
        Place droplets at the starting node (origin).
        """
        self.droplets = {idx: self.origin for idx in range(self.n_nodes)}

    def move(self, droplet_id: int) -> None:
        """
        Move a droplet to a neighboring node based on transition probabilities.

        Args:
            droplet_id (int): The droplet ID.
        """
        current_node = self.droplets[droplet_id]
        if current_node == self.destination:
            return

        transition_probs = self.transition_probabilities(current_node)
        neighbors = list(transition_probs.keys())
        random.shuffle(neighbors)  # Shuffle for fairness among neighbors

        for neighbor in neighbors:
            prob = transition_probs[neighbor]
            if random.random() <= prob:  # Move with probability
                self.droplets[droplet_id] = neighbor
                self.erosions[current_node] = self.erosion(current_node, neighbor)
                break

    def erode(self) -> None:
        """
        Apply erosion to nodes based on droplets' movements.
        """
        for node, erosion in self.erosions.items():
            if node == self.destination:
                continue

            self.graph.nodes[node]["altitude"] -= erosion

    def deposit_sediments(self) -> None:
        erosion_produced = sum(self.erosions.values())

        new_altitudes = {
            node: {
                "altitude": self.graph.nodes[node]["altitude"]
                + erosion_produced / (self.n_nodes - 1)
            }
            for node in self.graph.nodes()
        }
        new_altitudes[self.destination] = {"altitude": 0}

        nx.set_node_attributes(self.graph, new_altitudes)

    def run(self, seasons: int = 1) -> None:
        """
        Run the RFD algorithm for a specified number of seasons.

        Args:
            seasons (int): Number of seasons to simulate.
        """
        self.initialise_altitudes()
        self.initialise_weights()

        for idx in range(seasons):
            print(f"Season {idx+1} / {seasons}", end="\r")

            self.place_droplets()

            n = 0
            while True:
                for droplet_id in self.droplets:
                    self.move(droplet_id)

                self.erode()
                self.deposit_sediments()

                if n % self.climbing_param_increment_every == 0:
                    self.climbing_param += self.climbing_param_increment

                n += 1

                if len(set(self.droplets.values())) == 1 or n == self.n_iterations:
                    break

        self.find_optimal_altitude_path()

    def find_optimal_altitude_path(self) -> list[tuple[int, int]]:
        """
        Determine the optimal path from origin to destination based on altitude.

        Returns:
            list: The optimal path as a sequence of nodes.
        """
        current_node = self.origin
        self.optimal_altitude_path = [current_node]

        while current_node != self.destination:
            neighbors = [
                n
                for n in self.graph.neighbors(current_node)
                if n not in self.optimal_altitude_path
            ]

            if not neighbors:
                return

            next_node = min(neighbors, key=lambda n: self.graph.nodes[n]["altitude"])
            self.optimal_altitude_path.append(next_node)

            current_node = next_node

        return self.optimal_altitude_path

    def path_cost(self, path: list[Hashable]) -> float:
        """
        Calculate the total cost (sum of distances) of a given path.

        Args:
            path (list[Hashable]): The path as a sequence of nodes.

        Returns:
            float: The total cost of the path.
        """
        path_edges = list(zip(path[:-1], path[1:]))
        return sum(self.graph.edges[edge]["distance"] for edge in path_edges)

    def plot_graph(self) -> None:
        """
        Plot the graph with nodes colored by altitude and the optimal path highlighted.
        """
        altitudes = [
            self.graph.nodes[node].get("altitude") for node in self.graph.nodes()
        ]
        pos = {node: node for node in self.graph.nodes()}

        norm = colors.Normalize(vmin=min(altitudes), vmax=max(altitudes))
        cmap = cm.viridis

        # Plot graph with altitudes colors maps
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        nx.draw(
            self.graph,
            pos,
            node_color=altitudes,
            cmap=cmap,
            with_labels=True,
            node_size=500,
            font_size=10,
            edge_color="gray",
            ax=ax,
        )

        # Add the optimal altitude path
        optimal_altitude_path_edges = list(
            zip(self.optimal_altitude_path[:-1], self.optimal_altitude_path[1:])
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=optimal_altitude_path_edges,
            edge_color="red",
            width=2,
        )

        # Add the weights in the edges
        edge_labels = nx.get_edge_attributes(self.graph, "distance")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_color="blue", font_size=8
        )

        # Add the color bar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(altitudes)
        plt.colorbar(sm, ax=ax, label="Altitude")

        plt.title("Graph Nodes Colored by Altitude")

        plt.savefig("graph.png")

    def plot_optimal_path_altitude(self) -> None:
        """
        Plot the altitude profile along the optimal path.
        """
        y = [self.graph.nodes[e].get("altitude") for e in self.optimal_altitude_path]

        plt.figure(figsize=(8, 6))
        plt.plot(
            range(len(y)), y, marker="o", linestyle="-", color="blue", label="Altitude"
        )

        plt.xlabel("Step Along Path", fontsize=12)
        plt.ylabel("Altitude", fontsize=12)
        plt.title("Altitude Profile Along Optimal Path", fontsize=14)

        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

        plt.savefig("path_altitude.png")


if __name__ == "__main__":
    G = nx.grid_2d_graph(10, 10)
    rfd = RFD(graph=G, origin=(1, 1), destination=(5, 5))
    rfd.run(seasons=100)
    rfd.plot_graph()
    rfd.plot_optimal_path_altitude()
    print(f"Optimal altitude path: {rfd.optimal_altitude_path}")
