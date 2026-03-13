import osmnx as ox

G = ox.graph_from_place(
    "Central Coast, New South Wales, Australia",
    network_type="drive"
)

ox.save_graphml(G, "Dataset/central_coast_roads.graphml")