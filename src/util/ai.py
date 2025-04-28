# Best first search-------------------------------------------------------------------------------------------
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
    print(f"Expanding Node: {node} | h(n) = {heuristics[node]}, g(n) = {g}, f(n) = {g + heuristics[node]}")
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
            ao_star_with_g(child, graph, heuristics, solved, solution, g_values, g + heuristics[child])
            all_solved &= solved.get(child, False)
        if all_solved:
            solved[node] = True
    else:
        child = best_successors[0] if isinstance(best_successors, tuple) else best_successors
        ao_star_with_g(child, graph, heuristics, solved, solution, g_values, g + heuristics[child])
        if solved.get(child, False):
            solved[node] = True

    return heuristics[node]

# a star-------------------------------------------------------------------------------------------

import heapq

def astar(graph, heuristics, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristics[start], 0, start, [start]))
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        print(f"Visiting: {current}  , h(n) = {heuristics[current]}, g(n) = {g}, f(n) = {f}")

        if current == goal:
            return path, g

        for neighbor, weight in graph[current]:
            new_g = g + weight
            new_f = new_g + heuristics[neighbor]
            heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

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

import random

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
print(f"Best individual: {best_individual}, Fitness: {fitness(best_individual)}")

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

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

configs = [
    {"hidden_layer_sizes": (5,), "activation": 'relu'},
    {"hidden_layer_sizes": (10, 5), "activation": 'tanh'},
    {"hidden_layer_sizes": (20, 10, 5), "activation": 'logistic'}
]

for i, cfg in enumerate(configs):
    print(f"\n🔹 Configuration {i+1}: Hidden Layers = {cfg['hidden_layer_sizes']}, Activation = {cfg['activation']}")
    model = MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'], activation=cfg['activation'], max_iter=1000, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# random forest ----------------------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
