# Best first search-------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import permutations
import numpy as np
from queue import PriorityQueue
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import random
import heapq
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

heuristics = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 2,
    'E': 6,
    'F': 1
}


def best_first_search(start, goal):
    visited = set()
    queue = []

    heapq.heappush(queue, (heuristics[start], start))

    path = []

    while queue:
        h, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        path.append(current)
        print(f"Visiting: {current} (h={h})")

        if current == goal:
            print("Goal reached!")
            return path

        for neighbor in graph[current]:
            if neighbor not in visited:
                heapq.heappush(queue, (heuristics[neighbor], neighbor))

    return path


start_node = 'A'
goal_node = 'F'
result_path = best_first_search(start_node, goal_node)

print("\n🛣️ Path taken:", " → ".join(result_path))


# ao star----------------------------------------------------------------------------------------------------------------------------

def ao_star_with_g(node, graph, heuristics, solved, solution, g_values, g=0):
    print(
        f"Expanding Node: {node} | h(n) = {heuristics[node]}, g(n) = {g}, f(n) = {g + heuristics[node]}")
    g_values[node] = g

    if node in solved:
        return heuristics[node]

    if not graph[node]:
        solved[node] = True
        return heuristics[node]

    min_cost = float('inf')
    best_successors = None

    for successor in graph[node]:
        if isinstance(successor, tuple) and successor[1] == 'AND':
            children = successor[0]
            total_h = sum(heuristics[child] for child in children)
            cost = g + total_h
            if cost < min_cost:
                min_cost = cost
                best_successors = successor
        else:
            child = successor[0] if isinstance(successor, tuple) else successor
            cost = g + heuristics[child]
            if cost < min_cost:
                min_cost = cost
                best_successors = successor

    heuristics[node] = min_cost - g
    solution[node] = best_successors

    if isinstance(best_successors, tuple) and best_successors[1] == 'AND':
        all_solved = True
        for child in best_successors[0]:
            ao_star_with_g(child, graph, heuristics, solved,
                           solution, g_values, g + heuristics[child])
            all_solved &= solved.get(child, False)
        if all_solved:
            solved[node] = True
    else:
        child = best_successors[0] if isinstance(
            best_successors, tuple) else best_successors
        ao_star_with_g(child, graph, heuristics, solved,
                       solution, g_values, g + heuristics[child])
        if solved.get(child, False):
            solved[node] = True

    return heuristics[node]

# a star-------------------------------------------------------------------------------------------


def astar(graph, heuristics, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristics[start], 0, start, [start]))

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        print(
            f"Visiting: {current}  , h(n) = {heuristics[current]}, g(n) = {g}, f(n) = {f}")

        if current == goal:
            return path, g

        for neighbor, weight in graph[current]:
            new_g = g + weight
            new_f = new_g + heuristics[neighbor]
            heapq.heappush(
                open_set, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')


graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 1), ('E', 4)],
    'C': [('F', 2)],
    'D': [],
    'E': [],
    'F': []
}

heuristics = {
    'A': 5,
    'B': 3,
    'C': 4,
    'D': 2,
    'E': 6,
    'F': 0
}

start_node = 'A'
goal_node = 'F'

path, cost = astar(graph, heuristics, start_node, goal_node)

if path:
    print("\n✅ Final Path:", " -> ".join(path))
    print("🧮 Total Cost:", cost)

# genetic algo-----------------------------------------------------------------------------


def fitness(x):
    return x**2 - 3*x + 2


def mutate(x):
    return x + random.uniform(-0.1, 0.1)


def crossover(x, y):
    return (x + y) / 2


population = [random.uniform(-10, 10) for _ in range(10)]

for generation in range(100):
    population = sorted(population, key=lambda x: fitness(x))
    new_population = population[:5]

    while len(new_population) < 10:
        parent1 = random.choice(new_population)
        parent2 = random.choice(new_population)

        child = crossover(parent1, parent2)
        if random.random() < 0.1:
            child = mutate(child)

        new_population.append(child)

    population = new_population

best_individual = min(population, key=lambda x: fitness(x))
print(
    f"Best individual: {best_individual}, Fitness: {fitness(best_individual)}")

# fuzzy logic -----------------------------------------------------------------------------


def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0


def fuzzify(x):
    return {
        'low': triangular_membership(x, 0, 0, 5),
        'high': triangular_membership(x, 5, 10, 10)
    }


def fuzzy_wash_time(dirt, grease):
    dirt_fuzz = fuzzify(dirt)
    grease_fuzz = fuzzify(grease)

    short_strength = min(dirt_fuzz['low'], grease_fuzz['low'])

    long_strength = max(dirt_fuzz['high'], grease_fuzz['high'])

    total_strength = short_strength + long_strength

    if total_strength == 0:
        return 0

    wash_time = (short_strength * 30 + long_strength * 90) / total_strength
    return wash_time


dirt = 7
grease = 8
time = fuzzy_wash_time(dirt, grease)
print(f"Recommended wash time: {time:.2f} minutes")

# ann ----------------------------------------------------------------------------


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

configs = [
    {"hidden_layer_sizes": (5,), "activation": 'relu'},
    {"hidden_layer_sizes": (10, 5), "activation": 'tanh'},
    {"hidden_layer_sizes": (20, 10, 5), "activation": 'logistic'}
]

for i, cfg in enumerate(configs):
    print(
        f"\n🔹 Configuration {i+1}: Hidden Layers = {cfg['hidden_layer_sizes']}, Activation = {cfg['activation']}")
    model = MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'],
                          activation=cfg['activation'], max_iter=1000, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# random forest ----------------------------------------------------------------------------


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("🌳 Decision Tree Accuracy:", dt_acc)
print("🌲 Random Forest Accuracy:", rf_acc)

# ------------------------------------------------------

# MY FILE


def fitness(x):
    return x * (1 - x)


def genetic_algorithm(generations=50, pop_size=10):
    population = [random.random() for _ in range(pop_size)]
    for _ in range(generations):
        population = sorted(population, key=fitness, reverse=True)[
            :pop_size // 2]
        population += [random.random() for _ in range(pop_size // 2)]
    best = max(population, key=fitness)
    print(f"Best solution: x = {best:.4f}, f(x) = {fitness(best):.4f}")


genetic_algorithm()

# ----------------------------------------------------------------------


def fitness(x):
    return x**2


def genetic_algorithm():
    population = [random.randint(-50, 50) for _ in range(10)]
    for _ in range(100):
        population = sorted(population, key=fitness, reverse=True)[
            :5]  # Keep top 5
        population += [x + random.randint(-5, 5) for x in population]  # Mutate
    print("Best solution:", max(population, key=fitness))


genetic_algorithm()


# --------------------------------------------


def best_first_search(graph, start, goal, heuristic):
    pq = PriorityQueue()
    pq.put((heuristic[start], start))
    visited = set()

    while not pq.empty():
        _, current = pq.get()
        if current in visited:
            continue
        visited.add(current)
        print("Visited:", current)
        if current == goal:
            return
        for neighbor in graph[current]:
            pq.put((heuristic[neighbor], neighbor))


graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
heuristic = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
best_first_search(graph, 'A', 'D', heuristic)


# ---------------------------------------------------


def astar_search(graph, start, goal, heuristic, cost):
    pq = PriorityQueue()
    pq.put((heuristic[start], start, 0, [start]))

    while not pq.empty():
        _, current, g, path = pq.get()
        if current == goal:
            print("Path found:", path)
            return
        for neighbor in graph[current]:
            new_g = g + cost[(current, neighbor)]
            pq.put((new_g + heuristic[neighbor],
                   neighbor, new_g, path + [neighbor]))


graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
heuristic = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
cost = {('A', 'B'): 1, ('A', 'C'): 2, ('B', 'D'): 4, ('C', 'D'): 3}
astar_search(graph, 'A', 'D', heuristic, cost)


def ao_star(node, graph, heuristic):
    if node not in graph or not graph[node]:
        return heuristic[node]
    best_path = min(graph[node], key=lambda path: sum(
        heuristic[n] for n in path))
    heuristic[node] = sum(heuristic[n] for n in best_path)
    for n in best_path:
        ao_star(n, graph, heuristic)


graph = {'A': [['B', 'C'], ['D']], 'B': [['E'], ['F']],
         'C': [['G']], 'D': [], 'E': [], 'F': [], 'G': []}
heuristic = {'A': 10, 'B': 4, 'C': 2, 'D': 3, 'E': 2, 'F': 3, 'G': 1}
ao_star('A', graph, heuristic)
print("Final Heuristics:", heuristic)

# --------------------------------------------


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, 1)
lr = 0.1

for _ in range(1000):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        output = 1 / (1 + np.exp(-z))
        error = y[i] - output
        weights += lr * error * X[i].reshape(-1, 1)
        bias += lr * error

for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    output = 1 / (1 + np.exp(-z))
    print(f"Input: {X[i]}, Output: {round(output[0])}")


# -----------------------------------------------------

def fuzzify(value):
    if value <= 3:
        return 'low'
    elif value <= 6:
        return 'medium'
    else:
        return 'high'


def infer(dirt, grease):
    if dirt == 'high' or grease == 'high':
        return 'long'
    elif dirt == 'medium' and grease == 'medium':
        return 'medium'
    else:
        return 'short'


def defuzzify(label):
    return {'short': 20, 'medium': 35, 'long': 50}[label]


def evaluate(dirt, grease):
    dirt_level = fuzzify(dirt)
    grease_level = fuzzify(grease)
    time = infer(dirt_level, grease_level)
    print(
        f"Dirt: {dirt} ({dirt_level}), Grease: {grease} ({grease_level}), Time: {defuzzify(time)} minutes")


evaluate(6, 4)


# -------------------------------------------------------------


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(X_test)))

# -----------------------------------------------------------------------------

# practical-file


# Best First Search


def best_first_search(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))
    while not pq.empty():
        cost, node = pq.get()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            print(f"Goal {goal} found!")
            return
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                pq.put((weight, neighbor))
    print("Goal not found.")

# A* Search


def a_star_search(graph, heuristic, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0 + heuristic[start], 0, start))
    while not pq.empty():
        f_cost, g_cost, node = pq.get()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            print(f"Goal {goal} found with cost {g_cost}!")
            return
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_g_cost = g_cost + weight
                new_f_cost = new_g_cost + heuristic[neighbor]
                pq.put((new_f_cost, new_g_cost, neighbor))
    print("Goal not found.")

# AO* Algorithm


class AONode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.cost = float('inf')


def ao_star(node, goal):
    if node.name == goal:
        print(f"Goal {goal} found!")
        return True
    for child in node.children:
        if ao_star(child, goal):
            return True
    return False


# Example graph
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 4), ('E', 2)],
    'C': [('F', 5)],
    'D': [],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}
heuristic = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 0}

# Example AO* tree
root = AONode('A')
child1 = AONode('B')
child2 = AONode('C')
child3 = AONode('D')
root.children = [child1, child2]
child1.children = [child3]

# Run algorithms
print("Best First Search:")
best_first_search(graph, 'A', 'G')

print("\nA* Search:")
a_star_search(graph, heuristic, 'A', 'G')

print("\nAO* Algorithm:")
ao_star(root, 'D')

# -----------------------------------------------------------------------


def solve_cryptarithmetic():
    for perm in permutations(range(10), 8):
        s, e, n, d, m, o, r, y = perm
        if s == 0 or m == 0:  # Leading digits cannot be zero
            continue
        send = s * 1000 + e * 100 + n * 10 + d
        more = m * 1000 + o * 100 + r * 10 + e
        money = m * 10000 + o * 1000 + n * 100 + e * 10 + y
        if send + more == money:
            print(f"SEND: {send}, MORE: {more}, MONEY: {money}")
            return


solve_cryptarithmetic()

# -----------------------------------------------------------------------------------


# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, 1)
learning_rate = 0.1

# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Train the model
for _ in range(1000):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        output = sigmoid(z)
        error = y[i] - output
        weights += learning_rate * error * X[i].reshape(-1, 1)
        bias += learning_rate * error

# Test the model
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    output = sigmoid(z)
    print(f"Input: {X[i]}, Output: {round(output[0])}")

# ---------------------------------------------------------------------


# Example data
data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Run K-Means for different values of k
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    print(f"\nFor k={k}:")
    print("Cluster Centers:", kmeans.cluster_centers_)
    print("Labels:", kmeans.labels_)

# --------------------------------------------------------------


# Example data
data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [
                2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Apply PCA
pca = PCA(n_components=2)
pca.fit(data)
print("Eigenvalues:", pca.explained_variance_)
print("Eigenvectors:", pca.components_)

# ---------------------------------------------------------------


# Load dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Train and evaluate SVM with different kernels
for kernel in ['linear', 'poly', 'rbf']:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(f"\nKernel: {kernel}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ============================================


# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Train and evaluate KNN for different values of k
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"\nk={k}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# -------------------------------------------------------------


# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Display tree
print(export_text(dt, feature_names=iris.feature_names))

# --------------------------------------------------------


# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(X_test)))


# ------------------------------------------------------


def fuzzify(value):
    if value <= 3:
        return 'low'
    elif value <= 6:
        return 'medium'
    else:
        return 'high'


def infer(dirt, grease):
    if dirt == 'high' or grease == 'high':
        return 'long'
    elif dirt == 'medium' and grease == 'medium':
        return 'medium'
    else:
        return 'short'


def defuzzify(label):
    return {'short': 20, 'medium': 35, 'long': 50}[label]


def evaluate(dirt, grease):
    dirt_level = fuzzify(dirt)
    grease_level = fuzzify(grease)
    time = infer(dirt_level, grease_level)
    print(
        f"Dirt: {dirt} ({dirt_level}), Grease: {grease} ({grease_level}), Time: {defuzzify(time)} minutes")


evaluate(6, 4)

# --------------------------------------------------------------


def fitness(x):
    return x**2


def genetic_algorithm(generations=50, pop_size=10):
    population = [random.randint(-50, 50) for _ in range(pop_size)]
    for _ in range(generations):
        population.sort(key=lambda x: -fitness(x))
        population = population[:pop_size // 2]
        population += [random.randint(-50, 50) for _ in range(pop_size // 2)]
    print("Best solution:", max(population, key=fitness))


genetic_algorithm()


# ---------------------------------------------------------------------

def nlp_program():
    print("Hi! I can help you with AI programs. Which program would you like?")
    print("1. Heuristic Algorithms\n2. Constraint Satisfaction\n3. ANN\n4. K-Means\n5. PCA\n6. SVM\n7. KNN\n8. Decision Tree\n9. Random Forest\n10. Fuzzy Logic\n11. Genetic Algorithm")
    choice = int(input("Enter your choice (1-11): "))

    if choice == 1:
        print("Running Heuristic Algorithms...")
        # Call Heuristic Algorithms code here
    elif choice == 2:
        print("Running Constraint Satisfaction...")
        # Call Constraint Satisfaction code here
    elif choice == 3:
        print("Running ANN...")
        # Call ANN code here
    elif choice == 4:
        print("Running K-Means...")
        # Call K-Means code here
    elif choice == 5:
        print("Running PCA...")
        # Call PCA code here
    elif choice == 6:
        print("Running SVM...")
        # Call SVM code here
    elif choice == 7:
        print("Running KNN...")
        # Call KNN code here
    elif choice == 8:
        print("Running Decision Tree...")
        # Call Decision Tree code here
    elif choice == 9:
        print("Running Random Forest...")
        # Call Random Forest code here
    elif choice == 10:
        print("Running Fuzzy Logic...")
        # Call Fuzzy Logic code here
    elif choice == 11:
        print("Running Genetic Algorithm...")
        # Call Genetic Algorithm code here
    else:
        print("Invalid choice!")


nlp_program()

# --------------------------------------------------------------------------

# Summary genetic algo 


'''import random

def fitness(x):

    return x * (1 - x)

def genetic_algorithm(generations=50, pop_size=10):

    population = [random.random() for _ in range(pop_size)]

    for _ in range(generations):

        population = sorted(population, key=fitness, reverse=True)[

            :pop_size // 2]

        population += [random.random() for _ in range(pop_size // 2)]

    best = max(population, key=fitness)

    print(f"Best solution: x = {best:.4f}, f(x) = {fitness(best):.4f}")

genetic_algorithm()

This Python code implements a simple genetic algorithm, a technique inspired by natural selection, to optimize the function fitness(x) = x * (1 - x). Let’s break it down step by step:

1. Fitness Function

python

def fitness(x):

return x * (1 - x)

This function evaluates how good a solution xx is.
It reaches a maximum when x=0.5x = 0.5, meaning the algorithm is trying to evolve towards that value.
def genetic_algorithm(generations=50, pop_size=10):

generations=50 → Number of iterations for evolving the population.
pop_size=10 → Size of the population (10 individuals).
3. Initializing the Population

python

population = [random.random() for _ in range(pop_size)]

Creates a random initial population of floating-point numbers between 0 and 1.
4. Evolution Process (Loop)

python

for _ in range(generations):

Runs the evolution process for 50 generations.
a. Selection of Best Individuals

python

population = sorted(population, key=fitness, reverse=True)[:pop_size // 2]

Sorts the population based on fitness(x), picking the top half (most fit individuals).
b. Generating New Offspring

python

population += [random.random() for _ in range(pop_size // 2)]

Fills the population back to 10 by introducing new random individuals.
This simulates mutation & new genetic diversity.
5. Finding the Best Solution

python

best = max(population, key=fitness)

After evolution, finds the best individual with the highest fitness.
6. Printing the Best Result

python

print(f"Best solution: x = {best:.4f}, f(x) = {fitness(best):.4f}")

Displays the best x value and the corresponding fitness score.
Summary

This genetic algorithm works as follows:

Start with a random population.
Evaluate fitness for each individual.
Select the best half of the population.
Introduce new random individuals.
Repeat for 50 generations.
Output the best solution found.
import random

def fitness(x):

    return x**2

def genetic_algorithm():

    population = [random.randint(-50, 50) for _ in range(10)]

    for _ in range(100):

        population = sorted(population, key=fitness, reverse=True)[:5]  # Keep top 5

        population += [x + random.randint(-5, 5) for x in population]  # Mutate

    print("Best solution:", max(population, key=fitness))

genetic_algorithm()

Sure! Let's go through this genetic algorithm step by step.

1. Importing the random Module

python

import random

This imports Python’s built-in random module to generate random numbers.
2. Defining the Fitness Function

python

def fitness(x):

return x**2

The fitness function evaluates the "quality" of a solution.
It returns the square of x, meaning higher values of x are considered more fit.
For example:
fitness(5) = 25
fitness(-5) = 25
Both positive and negative numbers have the same fitness.
3. Defining the Genetic Algorithm

python

def genetic_algorithm():

This function runs the genetic algorithm to evolve a population towards the best possible solution.
4. Initializing the Population

python

population = [random.randint(-50, 50) for _ in range(10)]

Creates a random population of 10 individuals.
Each individual is a number randomly chosen between -50 and 50.
Example Population: [23, -12, 45, -50, 17, 33, -42, 8, -25, 50]

5. Running the Evolution Process

python

for _ in range(100):

The algorithm repeats 100 generations to gradually evolve towards better solutions.
6. Selecting the Best Individuals

python

population = sorted(population, key=fitness, reverse=True)[:5]

Sorts the population in descending order based on fitness(x).
Keeps the top 5 most fit individuals.
This is natural selection, where only the strongest survive.
7. Creating New Mutated Offspring

python

population += [x + random.randint(-5, 5) for x in population]

Mutates existing individuals by adding a small random value (-5 to +5).
This introduces variation, allowing new possibilities.
Example Mutation Process:

Suppose x = 45, mutation could produce values like 48, 43, etc.
This prevents the algorithm from getting stuck in local optima.
8. Finding the Best Solution

python

print("Best solution:", max(population, key=fitness))

After 100 generations, finds the best individual with the highest fitness.
Prints the final result.
Summary

Start with random numbers.
Select the top 5 most fit numbers.
Mutate them to create new offspring.
Repeat for 100 generations.
Print the best solution.'''