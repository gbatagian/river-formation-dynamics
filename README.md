# River Formation Dynamics (RFD) Algorithm

## About

This project implements the River Formation Dynamics (RFD) algorithm to find optimal paths on a graph. 
Inspired by the natural process of river formation, the algorithm simulates water flow and erosion to reinforce paths 
from an origin node to a destination node. It is applied to a 2D grid graph and generates two output images:

1. **graph.png**: Visualizes the graph with nodes colored by altitude and the optimal path highlighted.
2. **path_altitude.png**: Displays the altitude profile along the optimal path.

By default, random weights are assigned to the edges of the graph for each execution.

## Setup

* Clone or download the repository and navigate to its root directory: `cd .../river-formation-dynamics`
* Ensure you have `pipenv` installed on your system. If not, you can install it with: `pip install pipenv`
* Create a new virtual environment: `pipenv shell`
* Install the dependencies: `pipenv install --ignore-pipfile`
* Run the project: `python rfd.py`