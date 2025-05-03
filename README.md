#  Graph Coloring with Ant Colony Optimization (ACO)

This project implements the **Graph Coloring Problem** using the **Ant Colony Optimization (ACO)** algorithm. Graph coloring is a well-known NP-hard problem where the goal is to assign colors to each vertex of a graph such that no two adjacent vertices share the same color — using the minimum number of colors.

##  Algorithm Used

**Ant Colony Optimization (ACO)** is a metaheuristic inspired by the foraging behavior of ants. In this project, it is used to explore coloring solutions by simulating artificial "ants" that construct feasible colorings based on pheromone trails and heuristic information.

##  Project Structure

- `AntColony.py`: Implementation of the ACO algorithm for graph coloring.
- `README.md`: Project documentation (you are here!).

##  How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ilyass-Dahaoui/graph-Coloring-with-ACO.git
   cd graph-Coloring-with-ACO
2. **Prepare your environment**:
Make sure you have Python 3 installed. You may use a virtual environment if desired.

##  Example Output

Node 0 --> Color 1  
Node 1 --> Color 2  
Node 2 --> Color 1  
...  
Chromatic Number: 4  
## Customization
You can modify to define your own graph structure:

python-repl

0 1

0 2

1 2

1 3
...
Each line represents an edge between two nodes.

## Parameters
You can tweak the ACO parameters (number of ants, pheromone importance, evaporation rate, etc.) directly in AntColony.py to experiment with different behaviors and performance.

## Background
ACO is particularly well-suited for combinatorial problems like graph coloring due to its ability to balance exploration and exploitation via pheromone trails and probabilistic decisions.

## License
This project is open-source and available under the MIT License.

## Contact
Developed by Ilyas Dahaoui
Feel free to reach out via LinkedIn or by creating an issue.



