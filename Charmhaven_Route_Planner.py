import os
import re
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import folium

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 1. Load Charmhaven suburb boundary (for map display only) ────────────────
print("Loading Charmhaven boundary...")
gdf_boundary = ox.geocode_to_gdf('R6036118', by_osmid=True)
poly = gdf_boundary.geometry.iloc[0]

# ── 2. Load all potholes — filter by Depot Zone and Outstanding status ────────
df = pd.read_csv("Dataset/potholes_clean.csv")

def parse_field(desc, field):
    m = re.search(rf'{field}: ([^<]+)', str(desc))
    return m.group(1).strip() if m else None

df['defect_code']   = df['description'].apply(lambda x: parse_field(x, 'Defect Code'))
df['num_potholes']  = df['description'].apply(lambda x: parse_field(x, 'Estimated Number of Potholes'))
df['defect_number'] = df['description'].apply(lambda x: parse_field(x, 'Defect Number'))
df['status']        = df['description'].apply(lambda x: parse_field(x, 'Status'))
df['depot_zone']    = df['description'].apply(lambda x: parse_field(x, 'Depot Zone'))

pts = df[
    (df['depot_zone'] == 'Charmhaven Zone') &
    (df['status'] == 'Outstanding')
].copy().reset_index(drop=True)
pts['num_potholes'] = pts['num_potholes'].astype(float)

print(f"Charmhaven Zone outstanding potholes: {len(pts)}")

# ── 3. Depot ──────────────────────────────────────────────────────────────────
DEPOT_LAT, DEPOT_LON = -33.236718, 151.493981

# ── 4. Load full Central Coast road network ───────────────────────────────────
print("Loading road network...")
G = ox.load_graphml("Dataset/central_coast_roads.graphml")
# Keep only the largest strongly connected component
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ── 5. Assign priority ────────────────────────────────────────────────────────
print("Querying POIs...")

# Road type via nearest edge
nearest_edges = ox.nearest_edges(G, pts['lon'].tolist(), pts['lat'].tolist(), return_dist=False)

def get_highway(u, v, k):
    hw = G.edges[u, v, k].get('highway', 'unclassified')
    return hw[0] if isinstance(hw, list) else hw

pts['highway'] = [get_highway(u, v, k) for u, v, k in nearest_edges]

# Query POIs using bounding box of all 99 potholes (covers full Charmhaven Zone)
lat_min, lat_max = pts['lat'].min() - 0.01, pts['lat'].max() + 0.01
lon_min, lon_max = pts['lon'].min() - 0.01, pts['lon'].max() + 0.01
from shapely.geometry import box as shapely_box
zone_bbox = shapely_box(lon_min, lat_min, lon_max, lat_max)

tags_hs = {'amenity': ['hospital', 'school', 'clinic']}
tags_sh = {'shop': ['mall', 'supermarket', 'department_store']}

try:
    gdf_hs = ox.features_from_polygon(zone_bbox, tags=tags_hs)
    hs_proj = gdf_hs.to_crs(epsg=32756)
    hs_coords = list(zip(
        hs_proj.geometry.centroid.to_crs(epsg=4326).y,
        hs_proj.geometry.centroid.to_crs(epsg=4326).x
    ))
    print(f"  Hospitals/Schools/Clinics found: {len(hs_coords)}")
except Exception as e:
    hs_coords = []
    print(f"  No hospitals/schools found: {e}")

try:
    gdf_sh = ox.features_from_polygon(zone_bbox, tags=tags_sh)
    sh_proj = gdf_sh.to_crs(epsg=32756)
    sh_coords = list(zip(
        sh_proj.geometry.centroid.to_crs(epsg=4326).y,
        sh_proj.geometry.centroid.to_crs(epsg=4326).x
    ))
    print(f"  Shopping found: {len(sh_coords)}")
except Exception as e:
    sh_coords = []
    print(f"  No shopping found: {e}")

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

PROXIMITY_M = 200

def nearest_dist(lat, lon, coord_list):
    if not coord_list:
        return float('inf')
    return min(haversine_m(lat, lon, clat, clon) for clat, clon in coord_list)

pts['dist_hs'] = pts.apply(lambda r: nearest_dist(r['lat'], r['lon'], hs_coords), axis=1)
pts['dist_sh'] = pts.apply(lambda r: nearest_dist(r['lat'], r['lon'], sh_coords), axis=1)

road_priority = {'motorway': 3, 'trunk': 3, 'primary': 3, 'secondary': 4}

def assign_priority(row):
    if row['dist_hs'] <= PROXIMITY_M:
        return 1
    if row['dist_sh'] <= PROXIMITY_M:
        return 2
    return road_priority.get(row['highway'], 5)

pts['priority'] = pts.apply(assign_priority, axis=1)

priority_label = {1: 'Hospital/School', 2: 'Shopping',
                  3: 'Primary Road', 4: 'Secondary Road', 5: 'Other Road'}

print("\nPriority distribution:")
for p, grp in pts.groupby('priority'):
    print(f"  P{p} ({priority_label[p]}): {len(grp)} potholes")

# ── 6. Service time ───────────────────────────────────────────────────────────
def service_time(n):
    if n <= 2:    return 30
    elif n <= 10: return 90
    else:         return 240

pts['service_min'] = pts['num_potholes'].apply(service_time)

# ── 7. Snap all points to nearest road nodes ──────────────────────────────────
print("\nSnapping to road nodes...")
all_lats = [DEPOT_LAT] + pts['lat'].tolist()
all_lons = [DEPOT_LON] + pts['lon'].tolist()
all_nodes = ox.nearest_nodes(G, all_lons, all_lats)
n_nodes = len(all_nodes)

# ── 8. Build distance matrix ──────────────────────────────────────────────────
print(f"Building {n_nodes}x{n_nodes} distance matrix...")
dist_matrix = np.full((n_nodes, n_nodes), np.inf)
np.fill_diagonal(dist_matrix, 0)

for i, node in enumerate(all_nodes):
    lengths = nx.single_source_dijkstra_path_length(G, node, weight='length')
    for j, target in enumerate(all_nodes):
        if target in lengths:
            dist_matrix[i][j] = lengths[target]

print("Distance matrix complete.")

# ── 9. Priority-constrained Nearest Neighbour routing ─────────────────────────
visited = [False] * len(pts)
route_idx = []
current = 0  # depot

for level in sorted(pts['priority'].unique()):
    candidates = [i for i in range(len(pts)) if pts.loc[i, 'priority'] == level]
    while True:
        unvisited = [i for i in candidates if not visited[i]]
        if not unvisited:
            break
        best = min(unvisited, key=lambda i: dist_matrix[current][i + 1])
        route_idx.append(best)
        visited[best] = True
        current = best + 1

node_seq = [0] + [i + 1 for i in route_idx] + [0]

# ── 10. Summary ───────────────────────────────────────────────────────────────
total_dist_m = sum(dist_matrix[node_seq[k]][node_seq[k+1]]
                   for k in range(len(node_seq) - 1))
total_service_min = pts['service_min'].sum()
travel_min = (total_dist_m / 1000) / 40 * 60

SEP = "-" * 50
print(f"\n{SEP}")
print(f"  ROUTE SUMMARY - Charmhaven Zone")
print(SEP)
print(f"  Total stops          : {len(route_idx)}")
print(f"  Total distance       : {total_dist_m/1000:.2f} km")
print(f"  Travel time          : {travel_min:.0f} min  ({travel_min/60:.1f} h)")
print(f"  Service time         : {total_service_min:.0f} min  ({total_service_min/60:.1f} h)")
print(f"  Total estimated time : {(travel_min+total_service_min)/60:.1f} h")
print(SEP)

print(f"\n  PRIORITY BREAKDOWN")
print(f"  {'-'*46}")
for p, grp in pts.groupby('priority'):
    pct = len(grp) / len(pts) * 100
    print(f"  P{p} ({priority_label[p]:<18}): {len(grp):>3} potholes  ({pct:.1f}%)")
print(f"  {'-'*46}")
print(f"  {'Total':<22}: {len(pts):>3} potholes")

print(f"\n  ROAD TYPE BREAKDOWN")
print(f"  {'-'*46}")
for hw, grp in pts.groupby('highway'):
    pct = len(grp) / len(pts) * 100
    print(f"  {hw:<26}: {len(grp):>3} potholes  ({pct:.1f}%)")
print(f"{SEP}\n")

# ── Save metrics to CSV ───────────────────────────────────────────────────────
# Route sequence with details
route_records = []
cumulative_dist = 0
cumulative_time = 0
for k, pts_i in enumerate(route_idx):
    row = pts.loc[pts_i]
    leg_dist = dist_matrix[node_seq[k]][node_seq[k+1]]
    leg_travel = (leg_dist / 1000) / 40 * 60
    cumulative_dist += leg_dist
    cumulative_time += leg_travel + row['service_min']
    route_records.append({
        'stop_order'       : k + 1,
        'defect_number'    : row['defect_number'],
        'defect_code'      : row['defect_code'],
        'priority'         : int(row['priority']),
        'priority_reason'  : priority_label[int(row['priority'])],
        'road_type'        : row['highway'],
        'num_potholes'     : int(row['num_potholes']),
        'service_time_min' : int(row['service_min']),
        'leg_distance_m'   : round(leg_dist, 1),
        'cumulative_dist_km': round(cumulative_dist / 1000, 2),
        'lat'              : row['lat'],
        'lon'              : row['lon'],
    })

route_df = pd.DataFrame(route_records)
route_df.to_csv("charmhaven_route_details.csv", index=False, encoding='utf-8-sig')

# Summary metrics
summary = {
    'zone'                  : ['Charmhaven'],
    'total_stops'           : [len(route_idx)],
    'total_distance_km'     : [round(total_dist_m / 1000, 2)],
    'travel_time_min'       : [round(travel_min, 1)],
    'service_time_min'      : [round(total_service_min, 1)],
    'total_time_hours'      : [round((travel_min + total_service_min) / 60, 1)],
    'p1_hospital_school'    : [len(pts[pts['priority'] == 1])],
    'p2_shopping'           : [len(pts[pts['priority'] == 2])],
    'p3_primary_road'       : [len(pts[pts['priority'] == 3])],
    'p4_secondary_road'     : [len(pts[pts['priority'] == 4])],
    'p5_other_road'         : [len(pts[pts['priority'] == 5])],
}
pd.DataFrame(summary).to_csv("charmhaven_route_summary.csv", index=False, encoding='utf-8-sig')

print("  Outputs saved:")
print("  - charmhaven_route.html          (interactive map)")
print("  - charmhaven_route_details.csv   (stop-by-stop breakdown)")
print("  - charmhaven_route_summary.csv   (overall metrics)")

# ── 11. Map with actual road paths ────────────────────────────────────────────
print("\nBuilding map...")
zone_center_lat = pts['lat'].mean()
zone_center_lon = pts['lon'].mean()
m = folium.Map(location=[zone_center_lat, zone_center_lon], zoom_start=11, tiles="OpenStreetMap")

priority_color = {1: 'red', 2: 'orange', 3: 'blue', 4: 'purple', 5: 'gray'}

# Draw Charmhaven Zone boundary (convex hull of all 99 potholes) in red
from shapely.geometry import MultiPoint
zone_hull = MultiPoint(list(zip(pts['lon'], pts['lat']))).convex_hull
zone_boundary_coords = [(y, x) for x, y in zone_hull.exterior.coords]
folium.PolyLine(
    zone_boundary_coords,
    color='red', weight=3, opacity=0.9,
    tooltip="Charmhaven Zone Boundary"
).add_to(m)

# Pre-compute per-leg distances and travel times for popup labels
leg_dist_list = [dist_matrix[node_seq[k]][node_seq[k+1]] for k in range(len(node_seq)-1)]
leg_travel_list = [(d / 1000) / 40 * 60 for d in leg_dist_list]

# Draw actual road paths with distance + travel time popup
for k in range(len(node_seq) - 1):
    node_a = all_nodes[node_seq[k]]
    node_b = all_nodes[node_seq[k + 1]]
    leg_d  = leg_dist_list[k]
    leg_t  = leg_travel_list[k]
    try:
        path_nodes = nx.shortest_path(G, node_a, node_b, weight='length')
        path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path_nodes]
        folium.PolyLine(
            path_coords,
            color='steelblue', weight=2.5, opacity=0.7,
            tooltip=f"Leg {k+1}: {leg_d:.0f} m  |  {leg_t:.1f} min travel"
        ).add_to(m)
    except nx.NetworkXNoPath:
        pass

# Draw depot
folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot (Charmhaven)",
    icon=folium.Icon(color='black', icon='home', prefix='fa')
).add_to(m)

# Draw pothole stops — popup includes service time and leg distance
cumul_dist = 0.0
cumul_time = 0.0
for stop_order, pts_i in enumerate(route_idx):
    row = pts.loc[pts_i]
    p = int(row['priority'])
    color = priority_color[p]
    leg_d = leg_dist_list[stop_order]
    leg_t = leg_travel_list[stop_order]
    cumul_dist += leg_d
    cumul_time += leg_t + row['service_min']
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>Stop #{stop_order+1}</b><br>"
            f"Defect: {row['defect_number']}<br>"
            f"Priority: P{p} – {priority_label[p]}<br>"
            f"Road type: {row['highway']}<br>"
            f"Potholes: {int(row['num_potholes'])}<br>"
            f"<b>Service time: {int(row['service_min'])} min</b><br>"
            f"<b>Travel from prev: {leg_d:.0f} m  ({leg_t:.1f} min)</b><br>"
            f"Cumulative distance: {cumul_dist/1000:.2f} km<br>"
            f"Cumulative time: {cumul_time:.0f} min",
            max_width=270
        ),
        tooltip=f"#{stop_order+1} P{p} | svc {int(row['service_min'])}min"
    ).add_to(m)

# Legend
legend_html = f"""
<div style="position:fixed;bottom:40px;left:40px;width:240px;
    z-index:9999;font-size:13px;background:white;border:2px solid grey;
    border-radius:8px;padding:10px;box-shadow:2px 2px 6px rgba(0,0,0,.3)">
  <b>Charmhaven Route</b><br>
  <span style="font-size:11px;color:#555;">Total: {total_dist_m/1000:.2f} km &nbsp;|&nbsp; {travel_min:.0f} min travel</span><br>
  <span style="font-size:11px;color:#555;">Service: {total_service_min:.0f} min &nbsp;|&nbsp; {(travel_min+total_service_min)/60:.1f} h total</span><br><br>
  <span style="color:red;">&#9679;</span> P1 – Hospital / School<br>
  <span style="color:orange;">&#9679;</span> P2 – Shopping Centre<br>
  <span style="color:blue;">&#9679;</span> P3 – Primary Road<br>
  <span style="color:purple;">&#9679;</span> P4 – Secondary Road<br>
  <span style="color:gray;">&#9679;</span> P5 – Other Road<br>
  <span style="color:black;">&#8962;</span> Depot<br>
  <span style="color:red;">&#9135;&#9135;&#9135;</span> Suburb Boundary
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

output = "charmhaven_route.html"
m.save(output)
print(f"Map saved: {output}")
