import requests
import json
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
import networkx as nx

def visualize_distance_matrix_graph(locations, distance_matrix, filename="distance_matrix_graph.png"):
    """
    Creates a graph visualization showing only the nodes and the direct distances between them.
    """
    plt.figure(figsize=(12, 12))
    G = nx.Graph()

    # Add nodes and weighted edges from the distance matrix
    edge_labels = {}
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            if distance_matrix[i][j] is not None:
                G.add_edge(i, j, weight=distance_matrix[i][j])
                edge_labels[(i, j)] = f"{distance_matrix[i][j]} km"

    pos = nx.spring_layout(G, seed=42) # Position nodes

    # Draw the graph components
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='green', node_size=800, label='Node 0 (Origin)')

    plt.title("Direct Routes and Distances Between All Nodes")
    plt.legend()
    plt.savefig(filename)
    print(f"\n✅ Graph of direct distances saved as {filename}")
    plt.close()

def visualize_complete_graph(locations, distance_matrix, optimized_path, filename="optimized_route_graph.png"):
    """
    Creates a complete graph visualization with all connections and highlights the optimized route.
    """
    plt.figure(figsize=(12, 12))
    G = nx.Graph()

    # Add nodes and weighted edges from the distance matrix
    edge_labels = {}
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            if distance_matrix[i][j] is not None:
                G.add_edge(i, j, weight=distance_matrix[i][j])
                edge_labels[(i, j)] = f"{distance_matrix[i][j]} km"

    pos = nx.spring_layout(G, seed=42) # Position nodes

    # Draw the complete graph
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    # Highlight the depot/origin and destination
    nx.draw_networkx_nodes(G, pos, nodelist=[optimized_path[0]], node_color='green', node_size=800, label='Origin')
    nx.draw_networkx_nodes(G, pos, nodelist=[optimized_path[-1]], node_color='red', node_size=800, label='Destination')

    # Highlight the optimized path
    route_edges = list(zip(optimized_path, optimized_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='black', width=2.5, arrows=True, arrowstyle='->', arrowsize=20)

    plt.title("Complete Network Graph with Optimized Route")
    plt.legend()
    plt.savefig(filename)
    print(f"\n✅ Complete graph visualization saved as {filename}")
    plt.close()


def get_distance_matrix_graphhopper(api_key, destinations):
    """
    Calculates the A-to-B distance for all pairs of locations.
    """
    endpoint = f"https://graphhopper.com/api/1/matrix?key={api_key}"
    
    points = [[loc[1], loc[0]] for loc in destinations] # lon, lat format
    payload = {"points": points, "out_arrays": ["distances"]}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        if "distances" in data:
            # Convert meters to km
            return [[round(d / 1000, 2) if d is not None else None for d in row] for row in data['distances']]
        else:
            return {"error": data.get('message', 'Failed to retrieve distance matrix.')}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error: {e}"}


def get_optimized_route_graphhopper(api_key, destinations):
    """
    Calculates the optimized route using the GraphHopper Optimization API.
    """
    if len(destinations) < 2:
        return {"error": "Please provide at least two destinations."}

    endpoint = f"https://graphhopper.com/api/1/vrp?key={api_key}"
    
    vehicle = {
        "vehicle_id": "my_vehicle",
        "start_address": {"location_id": f"loc_{0}", "lat": destinations[0][0], "lon": destinations[0][1]},
        "end_address": {"location_id": f"loc_{len(destinations) - 1}", "lat": destinations[-1][0], "lon": destinations[-1][1]}
    }

    services = [{
        "id": str(i + 1),
        "address": {"location_id": f"loc_{i+1}", "lat": lat, "lon": lon}
    } for i, (lat, lon) in enumerate(destinations[1:-1])]

    payload = {"vehicles": [vehicle], "services": services}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

        if "solution" in data:
            solution = data['solution']
            total_distance_km = round(solution['distance'] / 1000, 2)
            
            activities = solution['routes'][0]['activities']
            optimized_path = []
            for activity in activities:
                if activity['type'] == 'start': optimized_path.append(0)
                elif activity['type'] == 'end': optimized_path.append(len(destinations) - 1)
                else: optimized_path.append(int(activity['id']))
            
            leg_details = []
            for i in range(len(activities) - 1):
                from_node = optimized_path[i]
                to_node = optimized_path[i+1]
                distance_km = round(activities[i+1]['distance'] / 1000, 2)
                leg_details.append({"from_node": from_node, "to_node": to_node, "distance_km": distance_km, "summary": f"Node {from_node} -> Node {to_node}"})

            return {"total_distance_km": total_distance_km, "optimized_path": optimized_path, "legs": leg_details}
        else:
            return {"error": data.get('message', 'An unknown error occurred.')}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    load_dotenv()
    API_KEY = os.getenv("GRAPHHOPPER_API_KEY")

    if not API_KEY or API_KEY == "YOUR_GRAPHHOPPER_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE SET YOUR 'GRAPHHOPPER_API_KEY' IN THE .env FILE      !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        locations = [
            (19.0760, 72.8777),  # 0: Mumbai (Origin)
            (28.7041, 77.1025),  # 1: Delhi
            (12.9716, 77.5946),  # 2: Bangalore
            (22.5726, 88.3639)   # 3: Kolkata (Destination)
        ]

        print("--- Calculating All Direct Routes (Distance Matrix) ---")
        distance_matrix = get_distance_matrix_graphhopper(API_KEY, locations)
        if "error" in distance_matrix:
            print(f"Error calculating distance matrix: {distance_matrix['error']}")
        else:
            for i in range(len(locations)):
                for j in range(len(locations)):
                    if i != j:
                        print(f"  Direct distance from Node {i} to Node {j}: {distance_matrix[i][j]} km")
            
            # Generate the visualization of the distance matrix
            visualize_distance_matrix_graph(locations, distance_matrix)
        
        print("\n--- Calculating Multi-Stop Optimized Route ---")
        result = get_optimized_route_graphhopper(API_KEY, locations)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- Optimized Route Leg Details ---")
            for leg in result['legs']:
                print(f"  {leg['summary']}: {leg['distance_km']} km")
            
            print("---------------------------------")
            print(f"Total Optimized Route Distance: {result['total_distance_km']} km")
            print(f"Optimized path order (by node index): {result['optimized_path']}")
            
            # Generate the visualization using the distance matrix
            if "error" not in distance_matrix:
                visualize_complete_graph(locations, distance_matrix, result['optimized_path'])

