from train import Route, Package, Vehicle

# # --- Define a Custom Logistics Network ---

# # A set of named locations for our test scenario
# LOCATIONS = [
#     "Central Depot",
#     "North Hub",
#     "South Retail",
#     "East Industrial Park",
#     "West Residential"
# ]

# # The routes connecting the locations with specific distances
# ROUTES = [
#     Route("Central Depot", "North Hub", 12),
#     Route("Central Depot", "East Industrial Park", 20),
#     Route("North Hub", "West Residential", 15),
#     Route("East Industrial Park", "South Retail", 10),
#     Route("West Residential", "South Retail", 25)
# ]

# # --- Define the Initial State for the Test ---

# # A specific list of packages to be delivered
# CUSTOM_PACKAGES = [
#     Package(id=0, pickup_location="North Hub", delivery_location="South Retail", weight=8),
#     Package(id=1, pickup_location="East Industrial Park", delivery_location="West Residential", weight=15),
#     Package(id=2, pickup_location="Central Depot", delivery_location="West Residential", weight=5),
#     Package(id=3, pickup_location="South Retail", delivery_location="North Hub", weight=12),
# ]

# # A specific fleet of vehicles with their starting locations and capacities
# CUSTOM_VEHICLES = [
#     Vehicle(id=0, capacity=20, current_location="Central Depot", speed=1.0, cost_per_km=1.0),
#     Vehicle(id=1, capacity=30, current_location="North Hub", speed=1.0, cost_per_km=1.0),
# ]


from train import Route, Package, Vehicle


LOCATIONS = ['mumbai', 'chennai', 'delhi'] 
ROUTES = [Route(start_location='mumbai', end_location='chennai', distance=1233.45), Route(start_location='mumbai', end_location='delhi', distance=1403.45), Route(start_location='chennai', end_location='mumbai', distance=1230.31), Route(start_location='chennai', end_location='delhi', distance=2196.57), Route(start_location='delhi', end_location='mumbai', distance=1395.75), Route(start_location='delhi', end_location='chennai', distance=2185.43)] 
CUSTOM_PACKAGES = [Package(id=0, pickup_location='mumbai', delivery_location='chennai', weight=3, status=0), Package(id=1, pickup_location='mumbai', delivery_location='delhi', weight=2, status=0)] 
CUSTOM_VEHICLES = [Vehicle(id=0, capacity=1000, current_location='mumbai', speed=1.0, cost_per_km=1.0, available_at_time=0)]