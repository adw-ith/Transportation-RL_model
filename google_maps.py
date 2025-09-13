import requests
import json
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt

def visualize_route(locations, optimized_path, filename="optimized_route_map.png"):
    """
    Creates and saves a plot of the locations and the optimized route.

    Args:
        locations (list of tuples): The original list of (lat, lng) coordinates.
        optimized_path (list of int): The optimized sequence of node indices.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(10, 10))
    
    # Extract coordinates for plotting
    lats = [loc[0] for loc in locations]
    lons = [loc[1] for loc in locations]

    # Plot all location nodes
    plt.scatter(lons, lats, c='blue', label='Destinations', s=100)

    # Annotate each node with its original index
    for i, (lat, lon) in enumerate(locations):
        plt.text(lon + 0.1, lat + 0.1, str(i), fontsize=12, ha='right')

    # Plot the origin node with a different marker
    origin_index = optimized_path[0]
    plt.scatter(locations[origin_index][1], locations[origin_index][0], c='green', marker='s', s=150, label='Origin', zorder=5)

    # Plot the destination node with a different marker
    dest_index = optimized_path[-1]
    plt.scatter(locations[dest_index][1], locations[dest_index][0], c='red', marker='X', s=150, label='Destination', zorder=5)

    # Draw arrows for the optimized route
    for i in range(len(optimized_path) - 1):
        start_node_idx = optimized_path[i]
        end_node_idx = optimized_path[i+1]
        
        start_lon, start_lat = locations[start_node_idx][1], locations[start_node_idx][0]
        end_lon, end_lat = locations[end_node_idx][1], locations[end_node_idx][0]
        
        plt.arrow(start_lon, start_lat, end_lon - start_lon, end_lat - start_lat,
                  color='black', length_includes_head=True, head_width=0.2, head_length=0.3)

    plt.title("Optimized Route Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)
    print(f"\n✅ Visualization saved as {filename}")
    plt.close()


def get_optimized_route(api_key, destinations):
    """
    Calculates the optimized route, total distance, and leg details.
    """
    if len(destinations) < 2:
        return {"error": "Please provide at least two destinations."}

    endpoint = "https://maps.googleapis.com/maps/api/directions/json"
    origin = f"{destinations[0][0]},{destinations[0][1]}"
    destination = f"{destinations[-1][0]},{destinations[-1][1]}"
    waypoints_list = [f"{lat},{lng}" for lat, lng in destinations[1:-1]]
    waypoints = "optimize:true|" + "|".join(waypoints_list) if waypoints_list else ""

    params = {'origin': origin, 'destination': destination, 'waypoints': waypoints, 'key': api_key}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            route = data['routes'][0]
            total_distance_meters = sum(leg['distance']['value'] for leg in route['legs'])
            total_distance_km = round(total_distance_meters / 1000, 2)

            # --- Construct the full optimized path and leg details ---
            original_indices_map = list(range(len(destinations)))
            origin_index = original_indices_map[0]
            destination_index = original_indices_map[-1]
            waypoint_indices = original_indices_map[1:-1]
            
            optimized_path = [origin_index]
            optimized_path.extend([waypoint_indices[i] for i in route.get('waypoint_order', [])])
            optimized_path.append(destination_index)
            
            leg_details = []
            for i, leg in enumerate(route['legs']):
                from_node = optimized_path[i]
                to_node = optimized_path[i+1]
                distance_km = round(leg['distance']['value'] / 1000, 2)
                leg_details.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "distance_km": distance_km,
                    "summary": f"Node {from_node} -> Node {to_node}"
                })

            return {
                "total_distance_km": total_distance_km,
                "optimized_path": optimized_path,
                "legs": leg_details
            }
        else:
            error_message = data.get('error_message', 'An unknown error occurred.')
            return {"error": f"API Error: {data['status']} - {error_message}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    if not API_KEY or API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE SET YOUR 'GOOGLE_MAPS_API_KEY' IN THE .env FILE   !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        locations = [
            (19.0760, 72.8777),  # 0: Mumbai (Origin)
            (28.7041, 77.1025),  # 1: Delhi
            (12.9716, 77.5946),  # 2: Bangalore
            (22.5726, 88.3639)   # 3: Kolkata (Destination)
        ]

        print(f"Calculating optimized route for {len(locations)} destinations...")
        result = get_optimized_route(API_KEY, locations)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- Optimized Route Details ---")
            for leg in result['legs']:
                print(f"  {leg['summary']}: {leg['distance_km']} km")
            
            print("---------------------------------")
            print(f"Total Optimized Route Distance: {result['total_distance_km']} km")
            print(f"Optimized path order (by node index): {result['optimized_path']}")
            
            # Generate the visualization
            visualize_route(locations, result['optimized_path'])

