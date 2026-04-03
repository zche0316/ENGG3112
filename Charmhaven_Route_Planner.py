import json
import hashlib
import html
import math
import os
import re

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPoint

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.environ.get("POTHOLE_DATASET", "Dataset/potholes_clean.csv")
ROAD_GRAPH_PATH = "Dataset/central_coast_roads.graphml"
ZONE_FILTER = os.environ.get("DEPOT_ZONE_FILTER", "Charmhaven Zone")
STATUS_FILTER = os.environ.get("STATUS_FILTER", "Outstanding")

DEPOT_LAT = -33.236718
DEPOT_LON = 151.493981
AVERAGE_SPEED_KMH = 40
SIMULATION_START_HOUR = 7.5
FUEL_EFFICIENCY_L_PER_100KM = 11.5
FUEL_PRICE_AUD_PER_L = 2.05

RANKING_WEIGHTS = {
    "Defect severity": 0.35,
    "Road importance": 0.25,
    "Response urgency": 0.15,
    "Traffic exposure": 0.15,
    "Route efficiency": 0.10,
}

ROAD_IMPORTANCE = {
    "motorway": 1.00,
    "trunk": 0.95,
    "primary": 0.90,
    "secondary": 0.75,
    "tertiary": 0.60,
    "residential": 0.40,
    "service": 0.35,
    "unclassified": 0.35,
}

ROAD_TYPE_LABEL = {
    "motorway": "Motorway / freight corridor",
    "trunk": "Trunk connector",
    "primary": "Primary connector road",
    "secondary": "Secondary connector road",
    "tertiary": "Tertiary local connector",
    "residential": "Residential street",
    "service": "Service access road",
    "unclassified": "Unclassified local street",
}

PRIORITY_COLORS = {
    "Immediate repair need": "#c62828",
    "High-impact connector repair": "#ef6c00",
    "Planned connector repair": "#1565c0",
    "Routine local repair": "#546e7a",
}

TRAFFIC_COLORS = {
    "Light": "#2a9d8f",
    "Moderate": "#f4a261",
    "Heavy": "#d62828",
}

PEAK_WINDOWS = {
    "school_peak": (7.0, 9.0, 14.5, 16.5),
    "work_peak": (7.0, 9.5, 16.0, 18.5),
    "freight_peak": (9.0, 15.0),
}


def parse_field(desc, field):
    match = re.search(rf"{re.escape(field)}: ([^<]+)", str(desc))
    return match.group(1).strip() if match else None


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_first_value(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


def stable_bucket(text, modulo):
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def travel_minutes(distance_m):
    return (distance_m / 1000) / AVERAGE_SPEED_KMH * 60


def fuel_litres(distance_m):
    return (distance_m / 1000) * (FUEL_EFFICIENCY_L_PER_100KM / 100)


def fuel_cost(distance_m):
    return fuel_litres(distance_m) * FUEL_PRICE_AUD_PER_L


def hour_to_label(hour_value):
    total_minutes = int(round(hour_value * 60)) % (24 * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def hour_in_window(hour_value, *bounds):
    if len(bounds) == 2:
        return bounds[0] <= hour_value <= bounds[1]
    if len(bounds) == 4:
        return bounds[0] <= hour_value <= bounds[1] or bounds[2] <= hour_value <= bounds[3]
    return False


def service_time_minutes(num_potholes):
    if num_potholes <= 2:
        return 30
    if num_potholes <= 10:
        return 90
    return 240


def format_duration(minutes):
    whole = int(round(minutes))
    if whole < 60:
        return f"{whole} min"
    hours = whole // 60
    mins = whole % 60
    return f"{hours} h {mins} min"


def extract_specific_location(row):
    road_name = row["street_name"] or row["dataset_name"] or "Unnamed road"
    suburb = row["suburb"] or "Unknown suburb"
    segment = row["road_segment"] or "segment not specified"
    return f"{road_name}, {suburb} ({segment})"


def road_importance_label(highway):
    highway = highway if highway in ROAD_TYPE_LABEL else "unclassified"
    return ROAD_TYPE_LABEL[highway]


def classify_priority_band(score_pct):
    if score_pct >= 75:
        return "Immediate repair need"
    if score_pct >= 60:
        return "High-impact connector repair"
    if score_pct >= 45:
        return "Planned connector repair"
    return "Routine local repair"


def build_priority_reason(row):
    reasons = [
        f"{int(row['num_potholes'])} pothole(s)",
        road_importance_label(row["highway"]),
        f"response target {int(row['response_time_days'])} day(s)",
    ]
    if row["school_zone_candidate"]:
        reasons.append("school-zone traffic exposure")
    if row["freight_corridor_candidate"]:
        reasons.append("freight corridor exposure")
    return ", ".join(reasons)


def precise_location_hint(row):
    return (
        f"Lat {row['lat']:.5f}, Lon {row['lon']:.5f} | "
        f"{row['specific_location']}"
    )


def midpoint(coords):
    if not coords:
        return None
    return coords[len(coords) // 2]


def simulate_leg_conditions(row, leg_distance_m, arrival_hour, stop_order):
    triggers = []
    factor = 1.0

    school_peak = row["school_zone_candidate"] and hour_in_window(arrival_hour, *PEAK_WINDOWS["school_peak"])
    work_peak = row["highway"] in {"motorway", "trunk", "primary", "secondary"} and hour_in_window(
        arrival_hour, *PEAK_WINDOWS["work_peak"]
    )
    freight_peak = row["freight_corridor_candidate"] and hour_in_window(arrival_hour, *PEAK_WINDOWS["freight_peak"])
    incident = stable_bucket(f"{row['defect_number']}-{stop_order}", 8) == 0

    if school_peak:
        triggers.append("school peak traffic")
        factor += 0.18
    if work_peak:
        triggers.append("work commute traffic")
        factor += 0.12
    if freight_peak:
        triggers.append("freight delivery traffic")
        factor += 0.10
    if incident:
        triggers.append("incident nearby")
        factor += 0.22

    base_minutes = travel_minutes(leg_distance_m)
    live_minutes = base_minutes * factor
    delay_minutes = live_minutes - base_minutes

    if factor >= 1.45:
        traffic_state = "Heavy"
        alert_level = "High"
    elif factor >= 1.20:
        traffic_state = "Moderate"
        alert_level = "Medium"
    else:
        traffic_state = "Light"
        alert_level = "Low"

    if not triggers:
        triggers = ["normal traffic conditions"]

    return {
        "arrival_time": hour_to_label(arrival_hour),
        "traffic_state": traffic_state,
        "alert_level": alert_level,
        "event_flags": ", ".join(triggers),
        "alert_message": f"{alert_level} alert: {', '.join(triggers)}",
        "base_travel_min": base_minutes,
        "simulated_travel_min": live_minutes,
        "traffic_delay_min": delay_minutes,
        "fuel_cost_aud": fuel_cost(leg_distance_m),
        "fuel_litres": fuel_litres(leg_distance_m),
    }


def build_data_input_panel(defect_locations_html, alert_rows_html, defect_selector_html):
    sample_path = html.escape(DATASET_PATH)
    return f"""
<div class="planner-panel planner-sidebar">
  <div class="panel-header">
    <div>
      <div class="panel-title">Defect Input And Route Context</div>
      <div class="panel-subtitle">Manual input, dataset source, precise defect locations, alerts, and legend.</div>
    </div>
  </div>

  <div class="planner-section">
    <div class="section-title">Manual Defect Input</div>
    <div class="helper-text">Enter a defect address directly in the HTML page.</div>
    <input class="planner-input planner-address-input" type="text" placeholder="Enter defect address">
    <div class="button-row">
      <button class="planner-btn planner-btn-primary">Add Defect</button>
    </div>
    <div class="helper-text">UI only: this simplified field currently shows the input interface but does not submit data.</div>
  </div>

  <div class="planner-section">
    <div class="section-title">Interactive Existing Defect Route Planner</div>
    <div class="helper-text">Select multiple existing defects, adjust weights, and generate an optimal route directly in the HTML page.</div>
    <div class="planner-weight-grid">
      <label class="weight-control">
        <span>Severity</span>
        <input id="weight-severity" type="range" min="0" max="100" value="35">
        <strong id="weight-severity-value">35%</strong>
      </label>
      <label class="weight-control">
        <span>Road importance</span>
        <input id="weight-road" type="range" min="0" max="100" value="25">
        <strong id="weight-road-value">25%</strong>
      </label>
      <label class="weight-control">
        <span>Urgency</span>
        <input id="weight-urgency" type="range" min="0" max="100" value="15">
        <strong id="weight-urgency-value">15%</strong>
      </label>
      <label class="weight-control">
        <span>Traffic exposure</span>
        <input id="weight-traffic" type="range" min="0" max="100" value="15">
        <strong id="weight-traffic-value">15%</strong>
      </label>
      <label class="weight-control">
        <span>Route efficiency</span>
        <input id="weight-efficiency" type="range" min="0" max="100" value="10">
        <strong id="weight-efficiency-value">10%</strong>
      </label>
    </div>
    <div class="button-row">
      <button class="planner-btn planner-btn-primary" id="generate-custom-route">Generate Route</button>
      <button class="planner-btn planner-btn-secondary" id="select-top-defects">Select Top 10</button>
    </div>
    <div class="selector-list" id="existing-defect-selector">
      {defect_selector_html}
    </div>
    <div id="custom-route-summary" class="data-card" style="margin-top:10px;">
      Choose at least 2 defects to generate a weighted route.
    </div>
  </div>

  <div class="planner-section">
    <div class="section-title">Current Dataset Source</div>
    <div class="data-card">
      <div><b>Current dataset:</b> {sample_path}</div>
      <div><b>Zone filter:</b> {html.escape(ZONE_FILTER)}</div>
      <div><b>Status filter:</b> {html.escape(STATUS_FILTER)}</div>
    </div>
    <div class="helper-text" style="margin-top:8px;">
      Expected CSV fields: `Name`, `description`, `lat`, `lon`. Scalable input is intended for 20+ defects.
    </div>
  </div>

  <div class="planner-section">
    <div class="section-title">Precise Defect Locations On Map</div>
    <div class="helper-text">Locations should use full addresses, road segments, or coordinates rather than vague place names.</div>
    <div class="location-list">
      {defect_locations_html}
    </div>
  </div>

  <div class="planner-section planner-alert-section">
    <div class="section-title">Traffic, Incident, And Peak-Hour Alerts</div>
    <div class="helper-text">Incidents plus school, work, and freight peak-hour effects are shown clearly.</div>
    <div class="alert-list">
      {alert_rows_html}
    </div>
  </div>

  <div class="planner-section">
    <div class="section-title">Clear Legend</div>
    <div class="legend-list">
      <div><span class="legend-dot" style="background:#c62828;"></span>Immediate repair need</div>
      <div><span class="legend-dot" style="background:#ef6c00;"></span>High-impact connector repair</div>
      <div><span class="legend-dot" style="background:#1565c0;"></span>Planned connector repair</div>
      <div><span class="legend-dot" style="background:#546e7a;"></span>Routine local street repair</div>
      <div><span class="legend-line" style="background:#2a9d8f;"></span>Light simulated traffic</div>
      <div><span class="legend-line" style="background:#f4a261;"></span>Moderate simulated traffic</div>
      <div><span class="legend-line" style="background:#d62828;"></span>Heavy simulated traffic</div>
      <div><span class="legend-home">⌂</span>Depot</div>
    </div>
  </div>
</div>
"""


print("Loading pothole dataset...")
df = pd.read_csv(DATASET_PATH)

df["dataset_name"] = df["Name"].fillna("") if "Name" in df.columns else ""
df["road_segment"] = df["description"].apply(lambda x: parse_field(x, "Road Segment Impacted"))
df["suburb"] = df["description"].apply(lambda x: parse_field(x, "Suburb"))
df["num_potholes"] = df["description"].apply(lambda x: as_float(parse_field(x, "Estimated Number of Potholes"), 1))
df["defect_number"] = df["description"].apply(lambda x: parse_field(x, "Defect Number"))
df["defect_code"] = df["description"].apply(lambda x: parse_field(x, "Defect Code"))
df["status"] = df["description"].apply(lambda x: parse_field(x, "Status"))
df["depot_zone"] = df["description"].apply(lambda x: parse_field(x, "Depot Zone"))
df["response_time_days"] = df["description"].apply(lambda x: as_float(parse_field(x, "Response Time"), 30))
df["location_code"] = df["description"].apply(lambda x: parse_field(x, "Location"))

pts = df[(df["depot_zone"] == ZONE_FILTER) & (df["status"] == STATUS_FILTER)].copy().reset_index(drop=True)
if pts.empty:
    raise ValueError(f"No defects found for zone='{ZONE_FILTER}' and status='{STATUS_FILTER}'.")

print(f"Selected defects: {len(pts)}")
print("Loading road network...")
G = ox.load_graphml(ROAD_GRAPH_PATH)
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

print("Attaching road context from local graph only...")
nearest_edges = ox.nearest_edges(G, pts["lon"].tolist(), pts["lat"].tolist(), return_dist=False)
pts["highway"] = [get_first_value(G.edges[u, v, k].get("highway"), "unclassified") for u, v, k in nearest_edges]
pts["street_name"] = [get_first_value(G.edges[u, v, k].get("name"), "") for u, v, k in nearest_edges]
pts["specific_location"] = pts.apply(extract_specific_location, axis=1)
pts["service_time_min"] = pts["num_potholes"].apply(service_time_minutes)
pts["road_importance_score"] = pts["highway"].map(lambda hw: ROAD_IMPORTANCE.get(hw, 0.35))
pts["severity_score"] = np.clip(pts["num_potholes"] / 12, 0, 1)
pts["urgency_score"] = 1 - np.clip(pts["response_time_days"] / 30, 0, 1)
pts["school_zone_candidate"] = pts.apply(
    lambda row: stable_bucket(f"{row['specific_location']}-school", 7) == 0,
    axis=1,
)
pts["freight_corridor_candidate"] = pts["highway"].isin(["motorway", "trunk", "primary"])
pts["traffic_exposure_score"] = (
    pts["school_zone_candidate"].astype(float) * 0.45
    + pts["freight_corridor_candidate"].astype(float) * 0.55
)

print("\nSnapping depot and defects to road nodes...")
all_lats = [DEPOT_LAT] + pts["lat"].tolist()
all_lons = [DEPOT_LON] + pts["lon"].tolist()
all_nodes = ox.nearest_nodes(G, all_lons, all_lats)
n_nodes = len(all_nodes)

print(f"Building {n_nodes}x{n_nodes} road distance matrix...")
dist_matrix = np.full((n_nodes, n_nodes), np.inf)
np.fill_diagonal(dist_matrix, 0)
for i, source_node in enumerate(all_nodes):
    lengths = nx.single_source_dijkstra_path_length(G, source_node, weight="length")
    for j, target_node in enumerate(all_nodes):
        if target_node in lengths:
            dist_matrix[i][j] = lengths[target_node]
print("Distance matrix complete.")

depot_leg_distances = pd.Series(dist_matrix[0][1:], index=pts.index).replace(np.inf, np.nan)
max_depot_leg = max(depot_leg_distances.max(skipna=True), 1)
pts["route_efficiency_score"] = (1 - (depot_leg_distances / max_depot_leg)).fillna(0)
pts["ranking_score"] = (
    pts["severity_score"] * RANKING_WEIGHTS["Defect severity"]
    + pts["road_importance_score"] * RANKING_WEIGHTS["Road importance"]
    + pts["urgency_score"] * RANKING_WEIGHTS["Response urgency"]
    + pts["traffic_exposure_score"] * RANKING_WEIGHTS["Traffic exposure"]
    + pts["route_efficiency_score"] * RANKING_WEIGHTS["Route efficiency"]
)
pts["ranking_score_pct"] = (pts["ranking_score"] * 100).round(1)
pts["priority_band"] = pts["ranking_score_pct"].apply(classify_priority_band)
pts["priority_reason"] = pts.apply(build_priority_reason, axis=1)

print("\nPriority distribution:")
for band, group in pts.groupby("priority_band"):
    print(f"  {band}: {len(group)} defects")

print("Constructing route...")
pts["route_node_index"] = np.arange(1, len(pts) + 1)
pts["band_rank"] = pts["priority_band"].map(
    {
        "Immediate repair need": 1,
        "High-impact connector repair": 2,
        "Planned connector repair": 3,
        "Routine local repair": 4,
    }
)

visited = [False] * len(pts)
route_idx = []
current = 0
for band_rank in sorted(pts["band_rank"].unique()):
    candidates = pts.index[pts["band_rank"] == band_rank].tolist()
    ordered = sorted(candidates, key=lambda idx: pts.loc[idx, "ranking_score"], reverse=True)
    while True:
        unvisited = [idx for idx in ordered if not visited[idx]]
        if not unvisited:
            break
        best = min(
            unvisited,
            key=lambda idx: (
                dist_matrix[current][idx + 1] - pts.loc[idx, "ranking_score"] * 250,
                dist_matrix[current][idx + 1],
            ),
        )
        route_idx.append(best)
        visited[best] = True
        current = best + 1

node_seq = [0] + [idx + 1 for idx in route_idx] + [0]
route_records = []
cumulative_dist_m = 0.0
cumulative_time_min = 0.0
simulated_total_travel_min = 0.0
total_fuel_cost_aud = 0.0

for leg_number, pts_i in enumerate(route_idx, start=1):
    row = pts.loc[pts_i]
    leg_distance_m = dist_matrix[node_seq[leg_number - 1]][node_seq[leg_number]]
    arrival_hour = SIMULATION_START_HOUR + cumulative_time_min / 60
    traffic = simulate_leg_conditions(row, leg_distance_m, arrival_hour, leg_number)
    cumulative_dist_m += leg_distance_m
    cumulative_time_min += traffic["simulated_travel_min"] + row["service_time_min"]
    simulated_total_travel_min += traffic["simulated_travel_min"]
    total_fuel_cost_aud += traffic["fuel_cost_aud"]

    route_records.append(
        {
            "stop_order": leg_number,
            "defect_number": row["defect_number"],
            "defect_code": row["defect_code"],
            "specific_location": row["specific_location"],
            "road_name": row["street_name"] or row["dataset_name"],
            "suburb": row["suburb"],
            "road_segment": row["road_segment"],
            "location_code": row["location_code"],
            "priority_band": row["priority_band"],
            "priority_reason": row["priority_reason"],
            "ranking_score_pct": row["ranking_score_pct"],
            "road_type": road_importance_label(row["highway"]),
            "num_potholes": int(row["num_potholes"]),
            "response_time_days": int(row["response_time_days"]),
            "service_time_min": int(row["service_time_min"]),
            "leg_distance_km": round(leg_distance_m / 1000, 2),
            "base_travel_min": round(traffic["base_travel_min"], 1),
            "simulated_travel_min": round(traffic["simulated_travel_min"], 1),
            "traffic_delay_min": round(traffic["traffic_delay_min"], 1),
            "traffic_state": traffic["traffic_state"],
            "alert_level": traffic["alert_level"],
            "event_flags": traffic["event_flags"],
            "alert_message": traffic["alert_message"],
            "fuel_litres": round(traffic["fuel_litres"], 2),
            "fuel_cost_aud": round(traffic["fuel_cost_aud"], 2),
            "arrival_time": traffic["arrival_time"],
            "cumulative_dist_km": round(cumulative_dist_m / 1000, 2),
            "cumulative_time_min": round(cumulative_time_min, 1),
            "lat": row["lat"],
            "lon": row["lon"],
        }
    )

route_df = pd.DataFrame(route_records)
route_df.to_csv("charmhaven_route_details.csv", index=False, encoding="utf-8-sig")

base_total_travel_min = route_df["base_travel_min"].sum()
total_service_min = route_df["service_time_min"].sum()

summary_df = pd.DataFrame(
    {
        "zone": [ZONE_FILTER],
        "dataset_path": [DATASET_PATH],
        "selected_defects": [len(route_df)],
        "total_distance_km": [round(route_df["leg_distance_km"].sum(), 2)],
        "base_travel_time_min": [round(base_total_travel_min, 1)],
        "simulated_travel_time_min": [round(simulated_total_travel_min, 1)],
        "traffic_delay_total_min": [round(simulated_total_travel_min - base_total_travel_min, 1)],
        "service_time_min": [round(total_service_min, 1)],
        "estimated_total_time_min": [round(simulated_total_travel_min + total_service_min, 1)],
        "estimated_total_fuel_cost_aud": [round(total_fuel_cost_aud, 2)],
    }
)
summary_df.to_csv("charmhaven_route_summary.csv", index=False, encoding="utf-8-sig")

print("\nRoute summary")
print("-" * 56)
print(f"Selected defects            : {len(route_df)}")
print(f"Base travel time            : {format_duration(base_total_travel_min)}")
print(f"Live traffic travel time    : {format_duration(simulated_total_travel_min)}")
print(f"Traffic delay total         : {format_duration(simulated_total_travel_min - base_total_travel_min)}")
print(f"Service time total          : {format_duration(total_service_min)}")
print(f"Estimated fuel cost         : AUD {total_fuel_cost_aud:.2f}")
print("-" * 56)

print("\nBuilding route map...")
zone_hull = MultiPoint(list(zip(route_df["lon"], route_df["lat"]))).convex_hull
map_center = [route_df["lat"].mean(), route_df["lon"].mean()]
m = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap")

if zone_hull.geom_type == "Polygon":
    hull_coords = [(y, x) for x, y in zone_hull.exterior.coords]
    folium.PolyLine(
        hull_coords,
        color="#8d99ae",
        weight=2,
        opacity=0.8,
        tooltip=f"{ZONE_FILTER} route extent",
    ).add_to(m)

route_lookup = {int(row["stop_order"]): row.to_dict() for _, row in route_df.iterrows()}
segment_rows_html = []

for leg_number in range(1, len(node_seq)):
    start_node = all_nodes[node_seq[leg_number - 1]]
    end_node = all_nodes[node_seq[leg_number]]
    stop_info = route_lookup.get(leg_number)
    if stop_info is None:
        continue
    try:
        path_nodes = nx.shortest_path(G, start_node, end_node, weight="length")
    except nx.NetworkXNoPath:
        continue

    path_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in path_nodes]
    line_color = TRAFFIC_COLORS[stop_info["traffic_state"]]
    folium.PolyLine(
        path_coords,
        color=line_color,
        weight=4,
        opacity=0.85,
        tooltip=(
            f"Segment {leg_number}: {stop_info['leg_distance_km']:.2f} km | "
            f"{stop_info['simulated_travel_min']:.1f} min | "
            f"AUD {stop_info['fuel_cost_aud']:.2f}"
        ),
    ).add_to(m)

    mid = midpoint(path_coords)
    if mid:
        folium.Marker(
            location=mid,
            icon=folium.DivIcon(
                html=(
                    f"<div style='font-size:10px;font-weight:700;color:#1d3557;"
                    f"background:#ffffffcc;border:1px solid #457b9d;border-radius:10px;"
                    f"padding:2px 6px;'>S{leg_number}</div>"
                )
            ),
        ).add_to(m)

for _, row in route_df.iterrows():
    band_color = PRIORITY_COLORS[row["priority_band"]]
    popup_html = (
        f"<b>Stop {int(row['stop_order'])}</b><br>"
        f"<b>Location:</b> {html.escape(row['specific_location'])}<br>"
        f"<b>Why ranked here:</b> {html.escape(row['priority_reason'])}<br>"
        f"<b>Priority band:</b> {html.escape(row['priority_band'])}<br>"
        f"<b>Ranking score:</b> {row['ranking_score_pct']:.1f}/100<br>"
        f"<b>Travel from previous stop:</b> {row['leg_distance_km']:.2f} km<br>"
        f"<b>Base driving time:</b> {row['base_travel_min']:.1f} min<br>"
        f"<b>Traffic-adjusted driving time:</b> {row['simulated_travel_min']:.1f} min<br>"
        f"<b>Traffic delay:</b> +{row['traffic_delay_min']:.1f} min<br>"
        f"<b>Service time on site:</b> {int(row['service_time_min'])} min<br>"
        f"<b>Fuel for this leg:</b> {row['fuel_litres']:.2f} L (AUD {row['fuel_cost_aud']:.2f})<br>"
        f"<b>Arrival time:</b> {row['arrival_time']}<br>"
        f"<b>Traffic state:</b> {row['traffic_state']}<br>"
        f"<b>Alert:</b> {html.escape(row['alert_message'])}"
    )
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=folium.Popup(popup_html, max_width=360),
        tooltip=(
            f"Stop {int(row['stop_order'])}: "
            f"{row['leg_distance_km']:.2f} km | "
            f"{row['simulated_travel_min']:.1f} min | "
            f"AUD {row['fuel_cost_aud']:.2f}"
        ),
        icon=folium.DivIcon(
            html=(
                f"<div style='width:28px;height:28px;border-radius:14px;"
                f"background:{band_color};color:white;text-align:center;line-height:28px;"
                f"font-weight:700;border:2px solid white;box-shadow:0 0 4px rgba(0,0,0,.4);'>"
                f"{int(row['stop_order'])}</div>"
            )
        ),
    ).add_to(m)
    segment_rows_html.append(
        "<tr>"
        f"<td>{int(row['stop_order'])}</td>"
        f"<td>{html.escape((row['road_name'] or 'Unnamed road')[:24])}</td>"
        f"<td>{row['leg_distance_km']:.2f} km</td>"
        f"<td>{row['simulated_travel_min']:.1f} min</td>"
        f"<td>AUD {row['fuel_cost_aud']:.2f}</td>"
        f"<td>{html.escape(row['priority_reason'][:40])}</td>"
        "</tr>"
    )

folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

weight_rows_html = "".join(
    f"<tr><td>{html.escape(name)}</td><td style='text-align:right;'>{weight * 100:.0f}%</td></tr>"
    for name, weight in RANKING_WEIGHTS.items()
)

top_alerts = route_df[route_df["alert_level"].isin(["High", "Medium"])].head(10)
alert_rows_html = "".join(
    f"<div><b>Stop {int(row['stop_order'])}</b> at {row['arrival_time']}: "
    f"{html.escape(row['alert_message'])}</div>"
    for _, row in top_alerts.iterrows()
)
if not alert_rows_html:
    alert_rows_html = "<div>No medium or high alerts in the current simulation window.</div>"

defect_locations_html = "".join(
    f"<div class='location-item'><span class='location-stop'>Stop {int(row['stop_order'])}</span> "
    f"{html.escape(precise_location_hint(row))}</div>"
    for _, row in route_df.iterrows()
)
if not defect_locations_html:
    defect_locations_html = "<div>No defect locations available.</div>"

interactive_df = pts.copy()
interactive_df["ui_label"] = interactive_df.apply(
    lambda row: f"{row['defect_number']} | {row['specific_location']}",
    axis=1,
)
interactive_df = interactive_df.sort_values(["ranking_score", "num_potholes"], ascending=[False, False]).reset_index(drop=True)
default_selected_ids = interactive_df.head(min(10, len(interactive_df))).index.tolist()
defect_selector_html = "".join(
    (
        f"<label class='selector-item'>"
        f"<input type='checkbox' class='existing-defect-checkbox' value='{idx}' "
        f"{'checked' if idx in default_selected_ids else ''}>"
        f"<span class='selector-text'><b>{html.escape(str(row['defect_number']))}</b><br>"
        f"{html.escape(str(row['specific_location']))}</span>"
        f"</label>"
    )
    for idx, row in interactive_df.iterrows()
)
if not defect_selector_html:
    defect_selector_html = "<div class='helper-text'>No existing defects available.</div>"

interactive_route_records = []
for idx, row in interactive_df.iterrows():
    interactive_route_records.append(
        {
            "id": idx,
            "defect_number": row["defect_number"],
            "specific_location": row["specific_location"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "severity_score": float(row["severity_score"]),
            "road_importance_score": float(row["road_importance_score"]),
            "urgency_score": float(row["urgency_score"]),
            "traffic_exposure_score": float(row["traffic_exposure_score"]),
            "route_efficiency_score": float(row["route_efficiency_score"]),
            "service_time_min": float(row["service_time_min"]),
            "num_potholes": int(row["num_potholes"]),
        }
    )
interactive_route_json = json.dumps(interactive_route_records)
depot_json = json.dumps({"lat": DEPOT_LAT, "lon": DEPOT_LON})

interactive_path_lookup = {}
for from_ui_idx, from_row in interactive_df.iterrows():
    from_node_seq_idx = int(from_row["route_node_index"])

    try:
        depot_to_nodes = nx.shortest_path(G, all_nodes[0], all_nodes[from_node_seq_idx], weight="length")
        interactive_path_lookup[f"depot-{from_ui_idx}"] = [
            [float(G.nodes[node]["y"]), float(G.nodes[node]["x"])] for node in depot_to_nodes
        ]
    except nx.NetworkXNoPath:
        interactive_path_lookup[f"depot-{from_ui_idx}"] = [
            [DEPOT_LAT, DEPOT_LON],
            [float(from_row["lat"]), float(from_row["lon"])],
        ]

    for to_ui_idx, to_row in interactive_df.iterrows():
        if from_ui_idx == to_ui_idx:
            continue
        to_node_seq_idx = int(to_row["route_node_index"])
        try:
            path_nodes = nx.shortest_path(
                G,
                all_nodes[from_node_seq_idx],
                all_nodes[to_node_seq_idx],
                weight="length",
            )
            interactive_path_lookup[f"{from_ui_idx}-{to_ui_idx}"] = [
                [float(G.nodes[node]["y"]), float(G.nodes[node]["x"])] for node in path_nodes
            ]
        except nx.NetworkXNoPath:
            interactive_path_lookup[f"{from_ui_idx}-{to_ui_idx}"] = [
                [float(from_row["lat"]), float(from_row["lon"])],
                [float(to_row["lat"]), float(to_row["lon"])],
            ]

interactive_path_lookup_json = json.dumps(interactive_path_lookup)

legend_html = f"""
<style>
  :root {{
    --panel-bg: rgba(255,255,255,0.96);
    --panel-border: #d8dee7;
    --panel-shadow: 0 14px 28px rgba(15, 23, 42, 0.18);
    --text-main: #1f2937;
    --text-soft: #5b6472;
    --accent: #1565c0;
    --accent-soft: #e8f1ff;
    --warn-soft: #fff4e8;
  }}
  .planner-panel {{
    position: fixed;
    z-index: 9999;
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    box-shadow: var(--panel-shadow);
    backdrop-filter: blur(8px);
    color: var(--text-main);
    font: 13px/1.45 "Segoe UI", "Helvetica Neue", sans-serif;
  }}
  .planner-sidebar {{
    top: 24px;
    left: 24px;
    width: 380px;
    max-height: calc(100vh - 48px);
    overflow: auto;
    padding: 16px;
  }}
  .planner-card-top {{
    top: 24px;
    right: 24px;
    width: 360px;
    padding: 16px;
  }}
  .planner-card-bottom {{
    right: 24px;
    bottom: 24px;
    width: 500px;
    max-height: 42vh;
    overflow: auto;
    padding: 16px;
  }}
  .panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 14px;
  }}
  .panel-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  .panel-subtitle {{
    color: var(--text-soft);
    font-size: 12px;
    margin-top: 2px;
  }}
  .planner-section {{
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #edf1f5;
  }}
  .planner-section:first-of-type {{
    margin-top: 0;
    padding-top: 0;
    border-top: none;
  }}
  .section-title {{
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .input-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .planner-input,
  .planner-textarea {{
    width: 100%;
    border: 1px solid #cfd8e3;
    border-radius: 10px;
    padding: 10px 12px;
    font: inherit;
    box-sizing: border-box;
    background: #fbfdff;
  }}
  .planner-address-input {{
    margin-top: 8px;
    font-size: 15px;
    padding: 14px 16px;
  }}
  .planner-textarea {{
    min-height: 84px;
    resize: vertical;
    margin-top: 8px;
  }}
  .button-row {{
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }}
  .planner-btn {{
    border: none;
    border-radius: 999px;
    padding: 9px 14px;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }}
  .planner-btn-primary {{
    color: white;
    background: linear-gradient(135deg, #1565c0, #1f7ae0);
  }}
  .planner-btn-secondary {{
    color: #2d3a4d;
    background: #ecf2f7;
  }}
  .helper-text {{
    font-size: 11px;
    color: var(--text-soft);
    margin-top: 6px;
  }}
  .data-card {{
    background: #f8fbfe;
    border: 1px solid #e3ebf4;
    border-radius: 12px;
    padding: 10px 12px;
  }}
  .planner-weight-grid {{
    display: grid;
    gap: 8px;
    margin-top: 8px;
  }}
  .weight-control {{
    display: grid;
    grid-template-columns: 108px 1fr 44px;
    gap: 8px;
    align-items: center;
    font-size: 12px;
  }}
  .weight-control input[type=range] {{
    width: 100%;
  }}
  .selector-list {{
    margin-top: 10px;
    max-height: 180px;
    overflow: auto;
    border: 1px solid #e3ebf4;
    border-radius: 12px;
    background: #fbfdff;
    padding: 8px;
  }}
  .selector-item {{
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 8px;
    border-radius: 10px;
  }}
  .selector-item:nth-child(odd) {{
    background: #f7fafc;
  }}
  .selector-item input {{
    margin-top: 2px;
  }}
  .selector-text {{
    font-size: 12px;
    color: var(--text-main);
  }}
  .location-list,
  .alert-list {{
    max-height: 210px;
    overflow: auto;
    padding-right: 4px;
  }}
  .location-item,
  .alert-list div {{
    padding: 8px 10px;
    border-radius: 10px;
    background: #f8fbfe;
    border: 1px solid #e7eef6;
    margin-bottom: 8px;
  }}
  .location-stop {{
    display: inline-block;
    font-weight: 700;
    color: white;
    background: #355070;
    border-radius: 999px;
    padding: 2px 8px;
    margin-right: 6px;
    font-size: 11px;
  }}
  .planner-alert-section .alert-list div {{
    background: var(--warn-soft);
    border-color: #f3d2a4;
  }}
  .legend-list div {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex: 0 0 10px;
  }}
  .legend-line {{
    width: 24px;
    height: 2px;
    display: inline-block;
    flex: 0 0 24px;
  }}
  .legend-home {{
    width: 24px;
    display: inline-block;
    text-align: center;
    font-weight: 700;
  }}
  .metric-table,
  .dashboard-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }}
  .metric-table th,
  .metric-table td,
  .dashboard-table th,
  .dashboard-table td {{
    padding: 8px 6px;
    border-bottom: 1px solid #edf1f5;
    text-align: left;
    vertical-align: top;
  }}
  .metric-table th:last-child,
  .metric-table td:last-child,
  .dashboard-table th:last-child,
  .dashboard-table td:last-child {{
    text-align: right;
  }}
  .dashboard-table th {{
    position: sticky;
    top: 0;
    background: rgba(255,255,255,0.98);
  }}
  @media (max-width: 1280px) {{
    .planner-sidebar {{ width: 340px; }}
    .planner-card-top {{ width: 320px; }}
    .planner-card-bottom {{ width: 420px; }}
  }}
</style>
{build_data_input_panel(defect_locations_html, alert_rows_html, defect_selector_html)}
<div class="planner-panel planner-card-top">
  <div class="panel-title">Decision Factors And Weights</div>
  <div class="panel-subtitle">The system shall display decision factors, weights, and transparent prioritisation logic.</div>
  <table class="metric-table">
    <tr><th align="left">Factor</th><th align="right">Weight</th></tr>
    {weight_rows_html}
  </table>
  <div class="helper-text" style="margin-top:10px;">
    Start time: {hour_to_label(SIMULATION_START_HOUR)}<br>
    Fuel assumption: {FUEL_EFFICIENCY_L_PER_100KM:.1f} L/100 km at AUD {FUEL_PRICE_AUD_PER_L:.2f}/L<br>
    Map only shows defects included in the final route.
  </div>
</div>
<div class="planner-panel planner-card-bottom">
  <div class="panel-title">Route Dashboard</div>
  <div class="panel-subtitle">Each row shows segment time, distance, cost, and justification in an interpretable format.</div>
  <table class="dashboard-table">
    <tr>
      <th align="left">Stop</th>
      <th align="left">Road</th>
      <th align="left">Leg</th>
      <th align="left">Drive</th>
      <th align="left">Fuel</th>
      <th align="left">Reason</th>
    </tr>
    {''.join(segment_rows_html)}
  </table>
</div>
<script>
  const existingDefects = {interactive_route_json};
  const depotPoint = {depot_json};
  const roadPathLookup = {interactive_path_lookup_json};
  let customRouteLayer = null;

  function getLeafletMap() {{
    for (const value of Object.values(window)) {{
      if (value && typeof value.eachLayer === 'function' && typeof value.setView === 'function') {{
        return value;
      }}
    }}
    return null;
  }}

  function haversineKm(a, b) {{
    const toRad = (deg) => deg * Math.PI / 180;
    const R = 6371;
    const dLat = toRad(b.lat - a.lat);
    const dLon = toRad(b.lon - a.lon);
    const lat1 = toRad(a.lat);
    const lat2 = toRad(b.lat);
    const x = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
  }}

  function pathDistanceKm(pathCoords) {{
    if (!pathCoords || pathCoords.length < 2) return 0;
    let total = 0;
    for (let i = 0; i < pathCoords.length - 1; i++) {{
      total += haversineKm(
        {{ lat: pathCoords[i][0], lon: pathCoords[i][1] }},
        {{ lat: pathCoords[i + 1][0], lon: pathCoords[i + 1][1] }}
      );
    }}
    return total;
  }}

  function getRoadPath(fromKey, toKey) {{
    return roadPathLookup[`${{fromKey}}-${{toKey}}`] || null;
  }}

  function getWeights() {{
    const weights = {{
      severity: Number(document.getElementById('weight-severity').value),
      road: Number(document.getElementById('weight-road').value),
      urgency: Number(document.getElementById('weight-urgency').value),
      traffic: Number(document.getElementById('weight-traffic').value),
      efficiency: Number(document.getElementById('weight-efficiency').value),
    }};
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
    return {{
      severity: weights.severity / total,
      road: weights.road / total,
      urgency: weights.urgency / total,
      traffic: weights.traffic / total,
      efficiency: weights.efficiency / total,
    }};
  }}

  function computePriorityScore(defect, weights) {{
    return (
      defect.severity_score * weights.severity +
      defect.road_importance_score * weights.road +
      defect.urgency_score * weights.urgency +
      defect.traffic_exposure_score * weights.traffic +
      defect.route_efficiency_score * weights.efficiency
    );
  }}

  function getSelectedDefects() {{
    const selectedIds = Array.from(document.querySelectorAll('.existing-defect-checkbox:checked'))
      .map((checkbox) => Number(checkbox.value));
    return existingDefects.filter((defect) => selectedIds.includes(defect.id));
  }}

  function buildWeightedRoute(selectedDefects, weights) {{
    const remaining = selectedDefects.map((defect) => ({{
      ...defect,
      priorityScore: computePriorityScore(defect, weights),
    }}));
    const route = [];
    let current = depotPoint;
    let currentKey = 'depot';
    let totalKm = 0;

    while (remaining.length) {{
      remaining.sort((a, b) => b.priorityScore - a.priorityScore);
      let bestIndex = 0;
      let bestValue = Infinity;
      remaining.forEach((defect, index) => {{
        const candidatePath = getRoadPath(currentKey, defect.id);
        const km = candidatePath ? pathDistanceKm(candidatePath) : haversineKm(current, defect);
        const value = km - defect.priorityScore * 8;
        if (value < bestValue) {{
          bestValue = value;
          bestIndex = index;
        }}
      }});
      const next = remaining.splice(bestIndex, 1)[0];
      next.pathCoords = getRoadPath(currentKey, next.id) || [[current.lat, current.lon], [next.lat, next.lon]];
      next.legKm = pathDistanceKm(next.pathCoords);
      totalKm += next.legKm;
      route.push(next);
      current = next;
      currentKey = next.id;
    }}

    return {{ route, totalKm }};
  }}

  function renderCustomRoute(result) {{
    const map = getLeafletMap();
    if (!map) return;
    if (customRouteLayer) {{
      customRouteLayer.remove();
    }}
    customRouteLayer = L.layerGroup().addTo(map);

    const latlngs = [[depotPoint.lat, depotPoint.lon]];
    result.route.forEach((defect, index) => {{
      if (defect.pathCoords && defect.pathCoords.length) {{
        defect.pathCoords.forEach((coord, coordIndex) => {{
          if (index === 0 && coordIndex === 0) return;
          latlngs.push(coord);
        }});
      }} else {{
        latlngs.push([defect.lat, defect.lon]);
      }}
      const marker = L.marker([defect.lat, defect.lon], {{
        icon: L.divIcon({{
          className: '',
          html: `<div style="width:30px;height:30px;border-radius:15px;background:#0f766e;color:white;text-align:center;line-height:30px;font-weight:700;border:2px solid white;box-shadow:0 4px 10px rgba(0,0,0,.25);">${{index + 1}}</div>`,
          iconSize: [30, 30],
          iconAnchor: [15, 15]
        }})
      }});
      marker.bindPopup(
        `<b>Stop ${{index + 1}}</b><br>` +
        `${{defect.defect_number}}<br>` +
        `${{defect.specific_location}}<br>` +
        `Weighted score: ${{(defect.priorityScore * 100).toFixed(1)}} / 100<br>` +
        `Travel from previous point: ${{defect.legKm.toFixed(2)}} km`
      );
      marker.addTo(customRouteLayer);
    }});

    result.route.forEach((defect) => {{
      L.polyline(defect.pathCoords || [[depotPoint.lat, depotPoint.lon], [defect.lat, defect.lon]], {{
        color: '#0f766e',
        weight: 5,
        opacity: 0.9,
        dashArray: '10 8'
      }}).addTo(customRouteLayer);
    }});

    map.fitBounds(latlngs, {{ padding: [50, 50] }});
  }}

  function updateCustomRouteSummary(result) {{
    const summary = document.getElementById('custom-route-summary');
    if (!result.route.length) {{
      summary.innerHTML = 'Choose at least 2 defects to generate a weighted route.';
      return;
    }}
    const rows = result.route.map((defect, index) =>
      `<div style="margin-top:6px;"><b>${{index + 1}}.</b> ${{defect.defect_number}} | ${{defect.legKm.toFixed(2)}} km | score ${{(defect.priorityScore * 100).toFixed(1)}} / 100</div>`
    ).join('');
    summary.innerHTML =
      `<b>Generated weighted route</b><br>` +
      `<span class="helper-text">Stops: ${{result.route.length}} | Approx distance: ${{result.totalKm.toFixed(2)}} km</span>` +
      rows;
  }}

  function syncWeightLabels() {{
    const pairs = [
      ['weight-severity', 'weight-severity-value'],
      ['weight-road', 'weight-road-value'],
      ['weight-urgency', 'weight-urgency-value'],
      ['weight-traffic', 'weight-traffic-value'],
      ['weight-efficiency', 'weight-efficiency-value'],
    ];
    pairs.forEach(([inputId, labelId]) => {{
      const input = document.getElementById(inputId);
      const label = document.getElementById(labelId);
      if (input && label) {{
        label.textContent = `${{input.value}}%`;
      }}
    }});
  }}

  function generateCustomRoute() {{
    const selected = getSelectedDefects();
    const summary = document.getElementById('custom-route-summary');
    if (selected.length < 2) {{
      summary.innerHTML = 'Please select at least 2 existing defects.';
      return;
    }}
    const result = buildWeightedRoute(selected, getWeights());
    updateCustomRouteSummary(result);
    renderCustomRoute(result);
  }}

  document.addEventListener('DOMContentLoaded', () => {{
    syncWeightLabels();
    document.querySelectorAll('.weight-control input[type=range]').forEach((input) => {{
      input.addEventListener('input', syncWeightLabels);
    }});
    const generateButton = document.getElementById('generate-custom-route');
    if (generateButton) {{
      generateButton.addEventListener('click', generateCustomRoute);
    }}
    const selectTopButton = document.getElementById('select-top-defects');
    if (selectTopButton) {{
      selectTopButton.addEventListener('click', () => {{
        const boxes = Array.from(document.querySelectorAll('.existing-defect-checkbox'));
        boxes.forEach((box, index) => {{
          box.checked = index < 10;
        }});
        generateCustomRoute();
      }});
    }}
    setTimeout(generateCustomRoute, 600);
  }});
</script>
"""

m.get_root().html.add_child(folium.Element(legend_html))

output = "charmhaven_route.html"
m.save(output)
print(f"Map saved: {output}")
