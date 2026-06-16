# =========================
# IMPORTS
# =========================
import pandas as pd
import heapq
import streamlit as st
import folium
from streamlit_folium import st_folium

# =========================
# SESSION STATE INITIALIZATION
# =========================
if "routes_computed" not in st.session_state:
    st.session_state.routes_computed = False

for key in [
    "g_route", "g_cost",
    "d_route", "d_dist",
    "a_route", "a_costs"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# NODE COORDINATES (SYNTHETIC)
# =========================
node_coords = {
    1: (12.9716, 77.5946),
    2: (12.9740, 77.5990),
    3: (12.9780, 77.6050),
    4: (12.9820, 77.6100),
    5: (12.9850, 77.6150),
    6: (12.9900, 77.6200),
    7: (12.9950, 77.6250),
    8: (13.0000, 77.6300)
}

# =========================
# LOAD DATASET & BUILD GRAPH
# =========================
df = pd.read_csv("data/processed/edges_clean.csv")

graph = {}
for _, row in df.iterrows():
    graph.setdefault(int(row["from_node"]), {})[
        int(row["to_node"])
    ] = int(row["cost"])

# =========================
# ALGORITHMS
# =========================
def greedy_to_target(graph, start, target):
    visited = {start}
    route = [start]
    total_cost = 0
    current = start

    while current != target:
        neighbors = graph.get(current, {})
        next_node = None
        min_cost = float("inf")

        for node, cost in neighbors.items():
            if node not in visited and cost < min_cost:
                next_node = node
                min_cost = cost

        if next_node is None:
            break

        route.append(next_node)
        total_cost += min_cost
        visited.add(next_node)
        current = next_node

    return route, total_cost


def dijkstra_with_path(graph, start):
    dist = {node: float("inf") for node in graph}
    prev = {node: None for node in graph}

    dist[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > dist[current_node]:
            continue

        for neighbor, cost in graph.get(current_node, {}).items():
            new_dist = current_dist + cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    return dist, prev


def heuristic(node, goal):
    return abs(node - goal)


def astar(graph, start, goal):
    open_set = [(0, start)]
    g_cost = {start: 0}
    prev = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for neighbor, cost in graph.get(current, {}).items():
            tentative = g_cost[current] + cost
            if neighbor not in g_cost or tentative < g_cost[neighbor]:
                g_cost[neighbor] = tentative
                f_cost = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost, neighbor))
                prev[neighbor] = current

    return g_cost, prev


def reconstruct_path(prev, start, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = prev.get(current)
    path.reverse()
    return path if path and path[0] == start else []

# =========================
# STREAMLIT UI
# =========================
st.title("Delivery Route Optimizer (Static Routing Demo)")

start_node = st.selectbox("Select Start Node (Depot)", list(graph.keys()))
end_node = st.selectbox("Select End Node (Delivery)", list(graph.keys()))

if start_node == end_node:
    st.warning("Start and End nodes must be different")

# =========================
# COMPUTE ROUTES
# =========================
if st.button("Compute Routes") and start_node != end_node:
    st.session_state.routes_computed = True

    st.session_state.g_route, st.session_state.g_cost = greedy_to_target(
        graph, start_node, end_node
    )

    st.session_state.d_dist, d_prev = dijkstra_with_path(graph, start_node)
    st.session_state.d_route = reconstruct_path(
        d_prev, start_node, end_node
    )

    st.session_state.a_costs, a_prev = astar(graph, start_node, end_node)
    st.session_state.a_route = reconstruct_path(
        a_prev, start_node, end_node
    )

# =========================
# MAP VISUALIZATION
# =========================
if st.session_state.routes_computed:
    m = folium.Map(location=node_coords[start_node], zoom_start=13)

    # Draw node points
    for node, coord in node_coords.items():
        folium.CircleMarker(
            location=coord,
            radius=6,
            popup=f"Node {node}",
            color="black",
            fill=True,
            fill_opacity=1
        ).add_to(m)

    # Start & End markers
    folium.Marker(
        location=node_coords[start_node],
        popup="START",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        location=node_coords[end_node],
        popup="END",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Draw routes (different thickness)
    def draw_route(route, color, label, weight):
        coords = [node_coords[n] for n in route]
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            tooltip=label
        ).add_to(m)

    draw_route(st.session_state.g_route, "red", "Greedy", 4)
    draw_route(st.session_state.d_route, "blue", "Dijkstra", 6)
    draw_route(st.session_state.a_route, "green", "A*", 2)

    st_folium(m, width=800, height=500)

    # Cost display
    st.subheader("Route Cost Comparison")
    st.write("🔴 Greedy Cost:", st.session_state.g_cost)
    st.write("🔵 Dijkstra Cost:", st.session_state.d_dist[end_node])
    st.write("🟢 A* Cost:", st.session_state.a_costs[end_node])
