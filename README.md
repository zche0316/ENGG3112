# ENGG3112: Road Defect Remediation and Route Optimization System

This project focuses on optimizing the maintenance operations for road defects in the Central Coast region, specifically targeting the Charmhaven area. By leveraging Geographic Information Systems (GIS) and graph theory algorithms, the system prioritizes road defects and plans efficient routes for maintenance crews.

## Project Scope

Prioritizing road defect remediation: Given a list of road defects based on an overall priority risk rating, what is the optimum path for a maintenance crew to follow to minimize the time to complete the works, whilst ensuring the highest order defects are addressed first?

This problem is further complicated by variables such as:
- The need for traffic control.
- Different types of maintenance treatments (e.g., temporary patching, semi-permanent repairs, or heavy patching).
- Avoiding scheduling works during peak traffic periods.
- Large geographic distances common in regional councils (some exceeding 10,000 km2).

## Features

- **Pothole Data Processing**: Cleans and parses raw pothole report data, filtering for specific zones and outstanding status.
- **Road Network Modeling**: Constructs a detailed road network model using OpenStreetMap data.
- **Priority Assessment**: Evaluates repair priority based on distance to Points of Interest (POIs) such as schools, hospitals, and shopping centers.
- **Route Planning**: Implements pathfinding algorithms to determine the most efficient sequence for maintenance repairs.
- **Interactive Visualization**: Generates Folium-based HTML maps to display road networks, pothole locations, and optimized routes.

## Project Structure

```text
.
├── Dataset/                  # Dataset directory
│   ├── potholes_clean.csv    # Pothole locations and status data
│   └── central_coast_roads.graphml # Central Coast road network graph data
├── Charmhaven_Route_Planner.py # Main script: Route planning and comprehensive analysis
├── Road_NetWork.py           # Script: Road network visualization
├── Road_defect.py            # Script: Pothole feature extraction and analysis
├── requirements.txt          # Project dependency list
└── README.md                 # Project documentation
```

## Getting Started

### 1. Prerequisites

Recommended Python version: 3.10+. It is advised to use a virtual environment:

```bash
# Create and activate virtual environment (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Windows
# python -m venv venv
# .\venv\Scripts\activate
```

### 2. Install Dependencies

Install the required libraries with specified version constraints:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

**Calculate maintenance routes and generate a map:**
```bash
python Charmhaven_Route_Planner.py
```
This will generate `NSW_pothole_map.html` in the root directory.

**Visualize the road network hierarchy:**
```bash
python Road_NetWork.py
```

## Data Sources and References

- **Road Network**: Extracted from OpenStreetMap via `osmnx`.
- **Pothole Data**: Data points extracted from below source in `Dataset/potholes_clean.csv`.
- **Council Reference**: [Central Coast Council Pothole Repairs and Map](https://www.centralcoast.nsw.gov.au/residents/roads/pothole-repairs-and-map)

## Dependencies

This project relies on the following core libraries:
- `osmnx` & `networkx`: Road network processing and graph analysis.
- `geopandas` & `shapely`: Spatial data operations.
- `folium`: Interactive map visualization.
- `pandas` & `numpy`: Data manipulation.
- `scikit-learn`: Support for clustering or advanced analysis.

---
*This project is developed as part of the ENGG3112 course.*
