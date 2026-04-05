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

# Historical crash casualty hotspots from NSW iROADS / Power BI 2024 data,
# manually digitised and reverse-geocoded. Schema: degree-of-casualty classification.
ACCIDENT_HOTSPOTS = [
    {"lat": -33.26839, "lon": 151.33731, "casualty": "Seriously Injured", "road": "Yarramalong Road, Wyong Creek"},
    # ── Wadalba / Alison / B70 W ─────────────────────
    {"lat": -33.26121, "lon": 151.42656, "casualty": "Seriously Injured", "road": "Brittania Drive, Watanobbi"},
    {"lat": -33.26284, "lon": 151.42999, "casualty": "Moderately Injured", "road": "Kitty Place, Watanobbi"},
    {"lat": -33.26397, "lon": 151.42883, "casualty": "Seriously Injured", "road": "Brittania Drive, Watanobbi"},
    {"lat": -33.26597, "lon": 151.42724, "casualty": "Moderately Injured", "road": "Harrington Close, Watanobbi"},
    {"lat": -33.26688, "lon": 151.43207, "casualty": "Seriously Injured", "road": "Golf Links Drive, Watanobbi"},
    {"lat": -33.26759, "lon": 151.43123, "casualty": "Moderately Injured", "road": "Brittania Drive, Watanobbi"},
    {"lat": -33.26858, "lon": 151.43558, "casualty": "Moderately Injured", "road": "Pacific Highway, Wyong"},
    {"lat": -33.27062, "lon": 151.43136, "casualty": "Moderately Injured", "road": "Pacific Highway, Watanobbi"},
    {"lat": -33.27436, "lon": 151.42964, "casualty": "Seriously Injured", "road": "Pacific Highway, Watanobbi"},  # 2 events
    {"lat": -33.27597, "lon": 151.42690, "casualty": "Moderately Injured", "road": "Cutler Drive, Watanobbi"},
    {"lat": -33.27229, "lon": 151.42539, "casualty": "Seriously Injured", "road": "Cutler Drive, Watanobbi"},
    {"lat": -33.27761, "lon": 151.42430, "casualty": "Minor/Other Injured", "road": "Chapman Avenue, Wyong"},
    {"lat": -33.27781, "lon": 151.42495, "casualty": "Moderately Injured", "road": "Crystal Crescent, Wyong"},
    # ── Wyong area ───────────────────────────────────
    {"lat": -33.28117, "lon": 151.42193, "casualty": "Moderately Injured", "road": "Anzac Avenue, Wyong"},
    # ── Jilliby / local roads ────────────────────────
    {"lat": -33.26974, "lon": 151.40560, "casualty": "Moderately Injured", "road": "Pacific Motorway, Wyong"},
    {"lat": -33.25947, "lon": 151.40358, "casualty": "Seriously Injured", "road": "Pacific Motorway, Alison"},  # 3 events
    {"lat": -33.26115, "lon": 151.39398, "casualty": "Seriously Injured", "road": "Jilliby Road, Alison"},
    {"lat": -33.25730, "lon": 151.39746, "casualty": "Minor/Other Injured", "road": "Hue Hue Road, Jilliby"},
    {"lat": -33.25312, "lon": 151.39735, "casualty": "Seriously Injured", "road": "Hue Hue Road, Jilliby"},
    {"lat": -33.26920, "lon": 151.44015, "casualty": "Seriously Injured", "road": "Pacific Highway, Wyong"},  # 2 events
    {"lat": -33.26939, "lon": 151.44145, "casualty": "Moderately Injured", "road": "Pacific Highway, Wyong"},
    {"lat": -33.26987, "lon": 151.44458, "casualty": "Seriously Injured", "road": "Pacific Highway, Wyong"},
    {"lat": -33.26694, "lon": 151.44536, "casualty": "Moderately Injured", "road": "Donaldson Street, Wyong"},
    {"lat": -33.27185, "lon": 151.37291, "casualty": "Seriously Injured", "road": "Yarramalong Road, Wyong Creek"},
    {"lat": -33.26293, "lon": 151.52131, "casualty": "Minor/Other Injured", "road": "Main Road, Toukley"},
    {"lat": -33.26262, "lon": 151.52349, "casualty": "Seriously Injured", "road": "Viewpoint Drive, Toukley"},
    {"lat": -33.26423, "lon": 151.53454, "casualty": "Moderately Injured", "road": "Dunleigh Street, Toukley"},
    # ── Noraville S / Budgewoi Rd south ──────────────
    {"lat": -33.26500, "lon": 151.54200, "casualty": "Seriously Injured", "road": "Victoria Avenue, Toukley"},
    {"lat": -33.26677, "lon": 151.54671, "casualty": "Moderately Injured", "road": "Heador Street, Toukley"},  # 2 events
    # ── Charmhaven / Hamlyn Terrace / B70 ────────────
    {"lat": -33.26578, "lon": 151.46510, "casualty": "Minor/Other Injured", "road": "Orchid Way, Wadalba"},
    {"lat": -33.26266, "lon": 151.46471, "casualty": "Seriously Injured", "road": "Minnesota Road, Hamlyn Terrace"},  # 5 events
    {"lat": -33.26189, "lon": 151.46486, "casualty": "Moderately Injured", "road": "Minnesota Road, Hamlyn Terrace"},
    {"lat": -33.26175, "lon": 151.45716, "casualty": "Killed", "road": "Pacific Highway, Wadalba"},  # 2 events
    {"lat": -33.26547, "lon": 151.45519, "casualty": "Seriously Injured", "road": "Wadalba"},
    {"lat": -33.25010, "lon": 151.46734, "casualty": "Moderately Injured", "road": "Warnervale Road, Hamlyn Terrace"},
    {"lat": -33.25137, "lon": 151.47574, "casualty": "Moderately Injured", "road": "Louisiana Road, Hamlyn Terrace"},
    {"lat": -33.25157, "lon": 151.47720, "casualty": "Moderately Injured", "road": "Warnervale Road, Hamlyn Terrace"},
    {"lat": -33.25197, "lon": 151.48938, "casualty": "Minor/Other Injured", "road": "B70, Woongarrah"},
    {"lat": -33.25270, "lon": 151.48874, "casualty": "Seriously Injured", "road": "Pacific Highway, Hamlyn Terrace"},  # 2 events
    {"lat": -33.25491, "lon": 151.50318, "casualty": "Moderately Injured", "road": "Wallarah Road, Gorokan"},
    {"lat": -33.25937, "lon": 151.51475, "casualty": "Moderately Injured", "road": "Wallarah Road, Gorokan"},
    {"lat": -33.25446, "lon": 151.49678, "casualty": "Moderately Injured", "road": "B70, Kanwal"},  # 2 events
    {"lat": -33.25593, "lon": 151.47480, "casualty": "Moderately Injured", "road": "Louisiana Road, Hamlyn Terrace"},
    {"lat": -33.25557, "lon": 151.48014, "casualty": "Moderately Injured", "road": "Primrose Drive, Hamlyn Terrace"},
    {"lat": -33.25701, "lon": 151.48149, "casualty": "Seriously Injured", "road": "Gladioli Avenue, Hamlyn Terrace"},
    {"lat": -33.26323, "lon": 151.47328, "casualty": "Moderately Injured", "road": "Louisiana Road, Hamlyn Terrace"},  # 2 events
    {"lat": -33.26418, "lon": 151.47310, "casualty": "Seriously Injured", "road": "Pacific Highway, Kanwal"},  # 3 events
    {"lat": -33.26194, "lon": 151.48259, "casualty": "Minor/Other Injured", "road": "Pacific Highway, Kanwal"},
    {"lat": -33.25394, "lon": 151.49344, "casualty": "Moderately Injured", "road": "Walker Avenue, Kanwal"},
    {"lat": -33.25868, "lon": 151.48842, "casualty": "Moderately Injured", "road": "Walker Avenue, Kanwal"},
    {"lat": -33.25742, "lon": 151.49575, "casualty": "Seriously Injured", "road": "Minnamurra Road, Gorokan"},
    {"lat": -33.26217, "lon": 151.49601, "casualty": "Seriously Injured", "road": "Marks Road, Gorokan"},
    {"lat": -33.27102, "lon": 151.45311, "casualty": "Moderately Injured", "road": "83, Wyong"},
    {"lat": -33.27227, "lon": 151.46101, "casualty": "Moderately Injured", "road": "Johns Road, Wadalba"},
    {"lat": -33.26625, "lon": 151.49000, "casualty": "Moderately Injured", "road": "Stanley Street, Wyongah"},
    {"lat": -33.26829, "lon": 151.55496, "casualty": "Seriously Injured", "road": "Main Road, Noraville"},
    {"lat": -33.26887, "lon": 151.55896, "casualty": "Moderately Injured", "road": "Main Road, Noraville"},
    {"lat": -33.26906, "lon": 151.56032, "casualty": "Moderately Injured", "road": "A49, Noraville"},
    {"lat": -33.26860, "lon": 151.56039, "casualty": "Minor/Other Injured", "road": "Budgewoi Road, Noraville"},
    {"lat": -33.27066, "lon": 151.55849, "casualty": "Seriously Injured", "road": "Brisbane Street, Noraville"},
    {"lat": -33.27136, "lon": 151.56009, "casualty": "Moderately Injured", "road": "Wilfred Barrett Drive, Norah Head"},
    {"lat": -33.27530, "lon": 151.55825, "casualty": "Moderately Injured", "road": "Wilfred Barrett Drive, Norah Head"},
    {"lat": -33.27936, "lon": 151.55559, "casualty": "Seriously Injured", "road": "Wilfred Barrett Drive, Noraville"},
    {"lat": -33.27841, "lon": 151.56436, "casualty": "Moderately Injured", "road": "Soldiers Point Drive, Norah Head"},
    {"lat": -33.24584, "lon": 151.39899, "casualty": "Seriously Injured", "road": "Hue Hue Road, Jilliby"},
    {"lat": -33.22685, "lon": 151.41524, "casualty": "Moderately Injured", "road": "Hue Hue Road, Jilliby"},
    {"lat": -33.22542, "lon": 151.41702, "casualty": "Seriously Injured", "road": "Sparks Road, Jilliby"},  # 3 events
    # ── Halloran area ────────────────────────────────
    {"lat": -33.22586, "lon": 151.42537, "casualty": "Moderately Injured", "road": "Pacific Motorway, Jilliby"},
    {"lat": -33.23065, "lon": 151.42060, "casualty": "Moderately Injured", "road": "Pacific Motorway, Jilliby"},
    {"lat": -33.23007, "lon": 151.42460, "casualty": "Seriously Injured", "road": "Sparks Road, Warnervale"},
    {"lat": -33.22910, "lon": 151.42296, "casualty": "Moderately Injured", "road": "Sparks Road, Warnervale"},
    {"lat": -33.23114, "lon": 151.42665, "casualty": "Seriously Injured", "road": "Sparks Road, Warnervale"},  # 3 events
    {"lat": -33.24011, "lon": 151.38083, "casualty": "Seriously Injured", "road": "Jilliby Road, Jilliby"},
    # ── Pacific Motorway (M1) area ───────────────────
    {"lat": -33.20752, "lon": 151.43148, "casualty": "Seriously Injured", "road": "Hue Hue Road, Jilliby"},
    {"lat": -33.21070, "lon": 151.43009, "casualty": "Moderately Injured", "road": "Hue Hue Road, Kiar"},
    {"lat": -33.21477, "lon": 151.42788, "casualty": "Moderately Injured", "road": "Hue Hue Road, Jilliby"},
    # ── Colongra / Scenic Drive ──────────────────────
    {"lat": -33.18973, "lon": 151.52895, "casualty": "Moderately Injured", "road": "Pacific Highway, Doyalson North"},  # 2 events
    {"lat": -33.19091, "lon": 151.52783, "casualty": "Seriously Injured", "road": "Pacific Highway, Doyalson"},
    {"lat": -33.19161, "lon": 151.52761, "casualty": "Moderately Injured", "road": "Pacific Highway, Doyalson"},  # 2 events
    {"lat": -33.19944, "lon": 151.52267, "casualty": "Moderately Injured", "road": "Scenic Drive, Doyalson"},  # 2 events
    # ── Blue Haven / San Remo ────────────────────────
    {"lat": -33.20578, "lon": 151.51639, "casualty": "Minor/Other Injured", "road": "Pacific Highway, Blue Haven"},
    {"lat": -33.20717, "lon": 151.49905, "casualty": "Seriously Injured", "road": "Blue Haven Way, Blue Haven"},  # 3 events
    {"lat": -33.20739, "lon": 151.49976, "casualty": "Moderately Injured", "road": "Roper Road, Blue Haven"},
    {"lat": -33.21374, "lon": 151.51219, "casualty": "Seriously Injured", "road": "Pacific Highway, San Remo"},  # 3 events
    {"lat": -33.21543, "lon": 151.50991, "casualty": "Moderately Injured", "road": "Pacific Highway, Blue Haven"},
    {"lat": -33.21545, "lon": 151.51385, "casualty": "Minor/Other Injured", "road": "Brava Avenue, San Remo"},
    {"lat": -33.21552, "lon": 151.52057, "casualty": "Moderately Injured", "road": "Goorama Avenue, San Remo"},
    {"lat": -33.21804, "lon": 151.51876, "casualty": "Moderately Injured", "road": "Liamena Avenue, San Remo"},
    {"lat": -33.21982, "lon": 151.51537, "casualty": "Seriously Injured", "road": "Emu Drive, San Remo"},
    {"lat": -33.21258, "lon": 151.52530, "casualty": "Seriously Injured", "road": "Eyre Crescent, San Remo"},
    # ── Budgewoi / Toukley / Noraville ───────────────
    {"lat": -33.22219, "lon": 151.53584, "casualty": "Seriously Injured", "road": "Scenic Drive, Buff Point"},
    {"lat": -33.21860, "lon": 151.53535, "casualty": "Seriously Injured", "road": "Scenic Drive, Colongra"},
    {"lat": -33.23230, "lon": 151.53026, "casualty": "Seriously Injured", "road": "Buff Point Avenue, Buff Point"},
    {"lat": -33.22910, "lon": 151.54206, "casualty": "Seriously Injured", "road": "Scenic Drive, Budgewoi"},
    {"lat": -33.23237, "lon": 151.54799, "casualty": "Moderately Injured", "road": "Scenic Drive, Budgewoi"},
    {"lat": -33.24874, "lon": 151.45826, "casualty": "Moderately Injured", "road": "Warnervale Road, Warnervale"},
    {"lat": -33.24856, "lon": 151.47723, "casualty": "Moderately Injured", "road": "Monarch Drive, Hamlyn Terrace"},
    {"lat": -33.24985, "lon": 151.48291, "casualty": "Minor/Other Injured", "road": "Sparks Road, Hamlyn Terrace"},
    {"lat": -33.24422, "lon": 151.47212, "casualty": "Seriously Injured", "road": "Hiawatha Road, Hamlyn Terrace"},
    {"lat": -33.24229, "lon": 151.49649, "casualty": "Moderately Injured", "road": "Pacific Highway, Charmhaven"},
    {"lat": -33.23872, "lon": 151.49932, "casualty": "Moderately Injured", "road": "83, Lake Haven"},  # 3 events
    {"lat": -33.23997, "lon": 151.50180, "casualty": "Minor/Other Injured", "road": "Lake Haven Drive, Lake Haven"},
    {"lat": -33.24474, "lon": 151.49738, "casualty": "Moderately Injured", "road": "Keswick Drive, Lake Haven"},
    {"lat": -33.24676, "lon": 151.49785, "casualty": "Moderately Injured", "road": "Gorokan Drive, Lake Haven"},
    {"lat": -33.24898, "lon": 151.50113, "casualty": "Moderately Injured", "road": "Gorokan Drive, Lake Haven"},
    {"lat": -33.24662, "lon": 151.50561, "casualty": "Moderately Injured", "road": "Gorokan Drive, Lake Haven"},
    {"lat": -33.22155, "lon": 151.50503, "casualty": "Moderately Injured", "road": "Pacific Highway, Charmhaven"},  # 2 events
    {"lat": -33.23005, "lon": 151.50295, "casualty": "Moderately Injured", "road": "Pacific Highway, Charmhaven"},  # 2 events
    {"lat": -33.23469, "lon": 151.50195, "casualty": "Killed", "road": "Pacific Highway, Charmhaven"},  # 2 events
    {"lat": -33.23524, "lon": 151.50168, "casualty": "Seriously Injured", "road": "Pacific Highway, Charmhaven"},
    {"lat": -33.23382, "lon": 151.55369, "casualty": "Moderately Injured", "road": "A49, Budgewoi"},  # 2 events
    {"lat": -33.23470, "lon": 151.55613, "casualty": "Seriously Injured", "road": "Scenic Drive, Budgewoi"},  # 2 events
    {"lat": -33.23994, "lon": 151.56541, "casualty": "Moderately Injured", "road": "A49, Budgewoi"},
    {"lat": -33.23487, "lon": 151.56819, "casualty": "Seriously Injured", "road": "Mimosa Road, Budgewoi"},
]

CASUALTY_DAILY_RATES = {
    "Killed":             0.05,
    "Seriously Injured":  0.20,
    "Moderately Injured": 0.35,
    "Minor/Other Injured":0.50,
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


def simulate_leg_conditions(row, leg_distance_m, leg_time_s, arrival_hour, stop_order):
    """leg_time_s: OSM-speed-based travel time in seconds (from time_matrix)."""
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

    # Base time from OSM speed data (road-type-aware), not a blanket 40 km/h assumption
    base_minutes = leg_time_s / 60
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


def build_data_input_panel(defect_locations_html, alert_rows_html, defect_selector_html, departure_time_str, incident_total):
    sample_path = html.escape(DATASET_PATH)
    return f"""
<div class="planner-panel planner-sidebar">
  <div class="panel-header">
    <div>
      <div class="panel-title">Defect Input And Route Context</div>
      <div class="panel-subtitle">Departure time, manual input, dataset source, precise defect locations, alerts, and legend.</div>
    </div>
    <button class="toggle-btn" onclick="togglePanel(this)" title="Collapse / expand">▾</button>
  </div>
  <div class="panel-body">

  <div class="planner-section" style="padding-top:0;border-top:none;">
    <div class="section-title">Departure Time</div>
    <div class="helper-text">Set the crew departure time. Traffic simulation and incident congestion update automatically.</div>
    <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
      <input id="departure-time-input" type="time" value="{departure_time_str}"
             class="planner-input" style="width:120px;font-size:16px;font-weight:700;padding:8px 10px;">
      <div style="font-size:12px;color:var(--text-soft);">
        <span id="incident-active-count" style="font-weight:700;color:#d32f2f;">0</span>
        of <span id="incident-total-count">{incident_total}</span> historical hotspots active today
      </div>
    </div>
  </div>

  <div class="planner-section">
    <div class="section-title">Today's Active Incidents</div>
    <div class="helper-text">
      Hotspot data covers all of 2024 ({incident_total} recorded locations).
      Set how many are treated as active today — higher-severity sites are selected first.
      Active incidents create congestion zones that increase travel time on nearby legs.
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
      <input id="incident-count-slider" type="range" min="0" max="{incident_total}" value="2"
             style="flex:1;accent-color:#d32f2f;">
      <strong id="incident-count-value" style="min-width:24px;color:#d32f2f;">2</strong>
      <button class="planner-btn planner-btn-secondary" style="padding:5px 12px;"
              onclick="rerandomizeIncidents()">Shuffle</button>
    </div>
    <div class="helper-text" style="margin-top:6px;" id="incident-impact-text">
      Drag slider then click Shuffle to activate a random subset weighted by casualty severity.
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
      <div><span class="legend-line" style="background:#2a9d8f;"></span>Light traffic</div>
      <div><span class="legend-line" style="background:#f4a261;"></span>Moderate traffic</div>
      <div><span class="legend-line" style="background:#d62828;"></span>Heavy traffic</div>
      <div><span class="legend-home">⌂</span>Depot</div>
      <div style="margin-top:8px;font-size:11px;font-weight:700;color:var(--text-soft);">Historical crash casualties</div>
      <div><span class="casualty-legend-dot" style="background:#e53935;"></span>Killed</div>
      <div><span class="casualty-legend-dot" style="background:#fb8c00;"></span>Seriously Injured</div>
      <div><span class="casualty-legend-dot" style="background:#37474f;"></span>Moderately Injured</div>
      <div><span class="casualty-legend-dot" style="background:#00acc1;"></span>Minor / Other Injured</div>
    </div>
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
print("Adding speed and travel-time attributes to edges...")
G = ox.add_edge_speeds(G)        # assigns speed (km/h) per highway type / maxspeed tag
G = ox.add_edge_travel_times(G)  # travel_time (s) = length / speed for each edge

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

print(f"Building {n_nodes}x{n_nodes} road distance and travel-time matrices...")
dist_matrix = np.full((n_nodes, n_nodes), np.inf)   # metres  – for fuel / distance reporting
time_matrix = np.full((n_nodes, n_nodes), np.inf)   # seconds – for routing decisions
np.fill_diagonal(dist_matrix, 0)
np.fill_diagonal(time_matrix, 0)
for i, source_node in enumerate(all_nodes):
    dist_lengths = nx.single_source_dijkstra_path_length(G, source_node, weight="length")
    time_lengths = nx.single_source_dijkstra_path_length(G, source_node, weight="travel_time")
    for j, target_node in enumerate(all_nodes):
        if target_node in dist_lengths:
            dist_matrix[i][j] = dist_lengths[target_node]
        if target_node in time_lengths:
            time_matrix[i][j] = time_lengths[target_node]
print("Distance and travel-time matrices complete.")

depot_leg_times = pd.Series(time_matrix[0][1:], index=pts.index).replace(np.inf, np.nan)
max_depot_leg_time = max(depot_leg_times.max(skipna=True), 1)
pts["route_efficiency_score"] = (1 - (depot_leg_times / max_depot_leg_time)).fillna(0)
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
                # Use travel-time (s) instead of raw distance.
                # ranking_score bonus: 1.0 score ≈ worth 60 s detour (≈ 250 m at 15 km/h local speed)
                time_matrix[current][idx + 1] - pts.loc[idx, "ranking_score"] * 60,
                time_matrix[current][idx + 1],
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
    leg_time_s = time_matrix[node_seq[leg_number - 1]][node_seq[leg_number]]
    arrival_hour = SIMULATION_START_HOUR + cumulative_time_min / 60
    traffic = simulate_leg_conditions(row, leg_distance_m, leg_time_s, arrival_hour, leg_number)
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

# ── Incident hotspot data for JS rendering ────────────────────────────────────
expected_daily_incidents = sum(
    CASUALTY_DAILY_RATES.get(h["casualty"], 0.35)
    for h in ACCIDENT_HOTSPOTS
)
accident_hotspots_json = json.dumps(ACCIDENT_HOTSPOTS)
_dh = int(SIMULATION_START_HOUR)
_dm = int((SIMULATION_START_HOUR - _dh) * 60)
departure_time_str = f"{_dh:02d}:{_dm:02d}"

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
  .toggle-btn {{
    background: none;
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    width: 28px;
    height: 28px;
    font-size: 14px;
    cursor: pointer;
    color: var(--text-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, color 0.15s;
  }}
  .toggle-btn:hover {{ background: var(--accent-soft); color: var(--accent); }}
  .panel-body {{
    overflow: hidden;
    transition: max-height 0.3s ease, opacity 0.25s ease;
    max-height: 3000px;
    opacity: 1;
  }}
  .panel-body.collapsed {{
    max-height: 0 !important;
    opacity: 0;
  }}
  .casualty-legend-dot {{
    width: 11px; height: 11px; border-radius: 50%;
    display: inline-block; flex: 0 0 11px;
    border: 1px solid rgba(0,0,0,0.18);
  }}
</style>
{build_data_input_panel(defect_locations_html, alert_rows_html, defect_selector_html, departure_time_str, len(ACCIDENT_HOTSPOTS))}
<div class="planner-panel planner-card-top">
  <div class="panel-header">
    <div>
      <div class="panel-title">Decision Factors And Weights</div>
      <div class="panel-subtitle">Prioritisation logic and crash hotspot summary.</div>
    </div>
    <button class="toggle-btn" onclick="togglePanel(this)" title="Collapse / expand">▾</button>
  </div>
  <div class="panel-body">
  <table class="metric-table">
    <tr><th align="left">Factor</th><th align="right">Weight</th></tr>
    {weight_rows_html}
  </table>
  <div class="helper-text" style="margin-top:10px;">
    Main route computed at: {hour_to_label(SIMULATION_START_HOUR)} — adjust departure in left panel.<br>
    Fuel: {FUEL_EFFICIENCY_L_PER_100KM:.1f} L/100 km @ AUD {FUEL_PRICE_AUD_PER_L:.2f}/L<br>
    Historical crash hotspots: <b>{len(ACCIDENT_HOTSPOTS)}</b> locations in area<br>
    Est. daily casualty events: <b>{expected_daily_incidents:.0f}</b><br>
    Map only shows defects included in the final route.
  </div>
  </div>
</div>
<div class="planner-panel planner-card-bottom">
  <div class="panel-header">
    <div>
      <div class="panel-title">Route Dashboard</div>
      <div class="panel-subtitle">Per-segment time, distance, cost, and justification.</div>
    </div>
    <button class="toggle-btn" onclick="togglePanel(this)" title="Collapse / expand">▾</button>
  </div>
  <div class="panel-body">
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
</div>
<script>
  const existingDefects = {interactive_route_json};
  const depotPoint = {depot_json};
  const roadPathLookup = {interactive_path_lookup_json};
  const accidentHotspots = {accident_hotspots_json};
  let customRouteLayer = null;
  let incidentLayer = null;
  let activeIncidents = [];   // subset of accidentHotspots active today

  // Severity weights: higher = more likely to be selected as "active"
  const CASUALTY_SEVERITY_WEIGHT = {{
    'Killed':             4,
    'Seriously Injured':  3,
    'Moderately Injured': 2,
    'Minor/Other Injured':1,
  }};

  // ── Map accessor ─────────────────────────────────────────────────────────────
  function getLeafletMap() {{
    for (const value of Object.values(window)) {{
      if (value && typeof value.eachLayer === 'function' && typeof value.setView === 'function') {{
        return value;
      }}
    }}
    return null;
  }}

  // ── Time helpers ─────────────────────────────────────────────────────────────
  function getDepartureHour() {{
    const input = document.getElementById('departure-time-input');
    if (!input || !input.value) return {SIMULATION_START_HOUR};
    const parts = input.value.split(':').map(Number);
    return parts[0] + parts[1] / 60;
  }}

  function hourToLabel(h) {{
    const totalMin = Math.round(((h % 24) + 24) % 24 * 60);
    const hh = String(Math.floor(totalMin / 60)).padStart(2, '0');
    const mm = String(totalMin % 60).padStart(2, '0');
    return hh + ':' + mm;
  }}

  function hourInPeak(h, ...bounds) {{
    if (bounds.length === 2) return h >= bounds[0] && h <= bounds[1];
    return (h >= bounds[0] && h <= bounds[1]) || (h >= bounds[2] && h <= bounds[3]);
  }}

  // ── Traffic delay factor at a given hour ─────────────────────────────────────
  function trafficFactor(arrivalHour) {{
    let f = 1.0;
    if (hourInPeak(arrivalHour, 7.0, 9.0, 14.5, 16.5)) f += 0.18;   // school peak
    if (hourInPeak(arrivalHour, 7.0, 9.5, 16.0, 18.5)) f += 0.12;   // work peak
    if (hourInPeak(arrivalHour, 9.0, 15.0))             f += 0.10;   // freight peak
    return f;
  }}

  function trafficLabel(factor) {{
    if (factor >= 1.38) return 'Heavy';
    if (factor >= 1.15) return 'Moderate';
    return 'Light';
  }}

  function trafficColor(factor) {{
    if (factor >= 1.38) return '#d62828';
    if (factor >= 1.15) return '#f4a261';
    return '#2a9d8f';
  }}

  // ── Distance helpers ─────────────────────────────────────────────────────────
  function haversineKm(a, b) {{
    const toRad = (d) => d * Math.PI / 180;
    const R = 6371;
    const dLat = toRad(b.lat - a.lat);
    const dLon = toRad(b.lon - a.lon);
    const x = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLon / 2) ** 2;
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
    return roadPathLookup[fromKey + '-' + toKey] || null;
  }}

  // ── Panel toggle ─────────────────────────────────────────────────────────────
  function togglePanel(btn) {{
    const body = btn.closest('.planner-panel').querySelector('.panel-body');
    if (!body) return;
    const isCollapsed = body.classList.toggle('collapsed');
    btn.textContent = isCollapsed ? '\u25b8' : '\u25be';
    btn.title = isCollapsed ? 'Expand' : 'Collapse';
  }}

  // ── Active incident management ────────────────────────────────────────────────

  // Weighted random selection of n hotspots (higher severity = more likely)
  function activateIncidentsRandom(n) {{
    if (n <= 0)  {{ activeIncidents = []; return; }}
    if (n >= accidentHotspots.length) {{ activeIncidents = [...accidentHotspots]; return; }}

    // Build cumulative weight table
    const weights = accidentHotspots.map(h => CASUALTY_SEVERITY_WEIGHT[h.casualty] || 1);
    const totalW  = weights.reduce((s, w) => s + w, 0);

    const selected = new Set();
    let attempts = 0;
    while (selected.size < n && attempts < n * 20) {{
      attempts++;
      let r = Math.random() * totalW;
      for (let i = 0; i < accidentHotspots.length; i++) {{
        r -= weights[i];
        if (r <= 0) {{ selected.add(i); break; }}
      }}
    }}
    activeIncidents = Array.from(selected).map(i => accidentHotspots[i]);
  }}

  // Congestion factor for a route leg based on proximity to active incidents
  // pathCoords: [[lat,lon], ...] — checks sampled points along the whole path
  function incidentCongestionForPath(pathCoords) {{
    if (!activeIncidents.length) return 1.0;
    const coords = pathCoords || [];
    // Sample up to 12 evenly spaced points along the path
    const step = Math.max(1, Math.floor(coords.length / 12));
    let maxFactor = 1.0;
    for (const inc of activeIncidents) {{
      const incPt = {{ lat: inc.lat, lon: inc.lon }};
      for (let i = 0; i < coords.length; i += step) {{
        const km = haversineKm({{ lat: coords[i][0], lon: coords[i][1] }}, incPt);
        let f = 1.0;
        if      (km < 0.30) f = 1.40;   // <300 m : heavy congestion
        else if (km < 0.70) f = 1.22;   // <700 m : moderate
        else if (km < 1.50) f = 1.09;   // <1.5 km: light spillback
        if (f > maxFactor) maxFactor = f;
        if (maxFactor >= 1.40) break;
      }}
      if (maxFactor >= 1.40) break;
    }}
    return maxFactor;
  }}

  // Slider + shuffle handler
  function rerandomizeIncidents() {{
    const n = Number(document.getElementById('incident-count-slider').value);
    activateIncidentsRandom(n);
    // Update counter label
    const countEl = document.getElementById('incident-active-count');
    if (countEl) countEl.textContent = activeIncidents.length;
    // Update impact text
    const el = document.getElementById('incident-impact-text');
    if (el) {{
      if (!activeIncidents.length) {{
        el.textContent = 'No active incidents — route uses time-of-day traffic only.';
      }} else {{
        const roads = [...new Set(activeIncidents.map(h => h.road.split(',')[0]))].slice(0, 3).join(', ');
        el.textContent = activeIncidents.length + ' active incident(s) near: ' + roads +
                         '. Nearby route legs show increased travel time.';
      }}
    }}
    renderIncidentLayer();
    generateCustomRoute();
  }}

  // ── Crash casualty hotspot rendering ─────────────────────────────────────────
  const CASUALTY_COLORS = {{
    'Killed':             '#e53935',
    'Seriously Injured':  '#fb8c00',
    'Moderately Injured': '#37474f',
    'Minor/Other Injured':'#00acc1',
  }};

  function renderIncidentLayer() {{
    const map = getLeafletMap();
    if (!map) return;
    if (incidentLayer) {{ incidentLayer.remove(); }}
    incidentLayer = L.layerGroup().addTo(map);
    const activeSet = new Set(activeIncidents.map(h => h.lat + ',' + h.lon));

    accidentHotspots.forEach((h) => {{
      const isActive = activeSet.has(h.lat + ',' + h.lon);
      const color  = CASUALTY_COLORS[h.casualty] || '#888';
      const radius = isActive ? 14 : 8;
      const opacity = isActive ? 1.0 : 0.25;
      const ring = isActive
        ? 'box-shadow:0 0 0 4px ' + color + '66,0 0 12px ' + color + '99,0 2px 6px rgba(0,0,0,.4);'
        : 'box-shadow:0 1px 3px rgba(0,0,0,.15);';

      const marker = L.marker([h.lat, h.lon], {{
        icon: L.divIcon({{
          className: '',
          html: '<div style="width:' + radius + 'px;height:' + radius + 'px;border-radius:50%;' +
                'background:' + color + ';opacity:' + opacity + ';' + ring + '"></div>',
          iconSize: [radius, radius],
          iconAnchor: [radius / 2, radius / 2],
        }})
      }});

      const statusNote = isActive
        ? '<span style="color:#d32f2f;font-weight:700;">\u26a0 ACTIVE TODAY — congestion zone active</span>'
        : '<span style="color:#888;">Historical record only (not active today)</span>';
      marker.bindPopup('<b>' + h.casualty + '</b><br>' + h.road + '<br>' + statusNote);
      marker.addTo(incidentLayer);
    }});

    const countEl = document.getElementById('incident-active-count');
    if (countEl) countEl.textContent = activeIncidents.length;
  }}

  // ── Weights & scoring ─────────────────────────────────────────────────────────
  function getWeights() {{
    const raw = {{
      severity:   Number(document.getElementById('weight-severity').value),
      road:       Number(document.getElementById('weight-road').value),
      urgency:    Number(document.getElementById('weight-urgency').value),
      traffic:    Number(document.getElementById('weight-traffic').value),
      efficiency: Number(document.getElementById('weight-efficiency').value),
    }};
    const total = Object.values(raw).reduce((s, v) => s + v, 0) || 1;
    return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / total]));
  }}

  function computePriorityScore(defect, weights) {{
    return (
      defect.severity_score        * weights.severity +
      defect.road_importance_score * weights.road +
      defect.urgency_score         * weights.urgency +
      defect.traffic_exposure_score * weights.traffic +
      defect.route_efficiency_score * weights.efficiency
    );
  }}

  function getSelectedDefects() {{
    const selectedIds = Array.from(document.querySelectorAll('.existing-defect-checkbox:checked'))
      .map((cb) => Number(cb.value));
    return existingDefects.filter((d) => selectedIds.includes(d.id));
  }}

  // ── Route building with departure-time-aware traffic simulation ───────────────
  function buildWeightedRoute(selectedDefects, weights) {{
    const departureHour = getDepartureHour();
    const remaining = selectedDefects.map((d) => ({{ ...d, priorityScore: computePriorityScore(d, weights) }}));
    const route = [];
    let current = depotPoint;
    let currentKey = 'depot';
    let totalKm = 0;
    let cumulativeMin = 0;

    while (remaining.length) {{
      remaining.sort((a, b) => b.priorityScore - a.priorityScore);
      let bestIndex = 0, bestValue = Infinity;
      remaining.forEach((defect, index) => {{
        const path = getRoadPath(currentKey, defect.id);
        const km = path ? pathDistanceKm(path) : haversineKm(current, defect);
        const val = km - defect.priorityScore * 8;
        if (val < bestValue) {{ bestValue = val; bestIndex = index; }}
      }});

      const next = remaining.splice(bestIndex, 1)[0];
      next.pathCoords = getRoadPath(currentKey, next.id) || [[current.lat, current.lon], [next.lat, next.lon]];
      next.legKm = pathDistanceKm(next.pathCoords);

      const arrivalHour = departureHour + cumulativeMin / 60;
      const tf  = trafficFactor(arrivalHour);
      const inc = incidentCongestionForPath(next.pathCoords);
      // Combine time-of-day traffic with incident congestion (multiplicative, capped at 2×)
      const combinedFactor = Math.min(tf * inc, 2.0);
      const baseDriveMin = next.legKm / 40 * 60;
      next.driveMin = baseDriveMin * combinedFactor;
      next.trafficFactor = combinedFactor;
      next.incidentDelay = inc > 1.0;
      next.trafficLabel = trafficLabel(combinedFactor);
      next.arrivalTime = hourToLabel(arrivalHour);
      cumulativeMin += next.driveMin + next.service_time_min;
      totalKm += next.legKm;
      route.push(next);
      current = next;
      currentKey = next.id;
    }}

    const totalH = Math.floor(cumulativeMin / 60);
    const totalM = Math.round(cumulativeMin % 60);
    return {{ route, totalKm, totalHours: totalH, totalMins: totalM }};
  }}

  // ── Map rendering ─────────────────────────────────────────────────────────────
  function renderCustomRoute(result) {{
    const map = getLeafletMap();
    if (!map) return;
    if (customRouteLayer) {{ customRouteLayer.remove(); }}
    customRouteLayer = L.layerGroup().addTo(map);

    const latlngs = [[depotPoint.lat, depotPoint.lon]];
    result.route.forEach((defect, index) => {{
      const lineColor = trafficColor(defect.trafficFactor);
      L.polyline(
        defect.pathCoords || [[depotPoint.lat, depotPoint.lon], [defect.lat, defect.lon]],
        {{ color: lineColor, weight: 5, opacity: 0.88, dashArray: '10 8' }}
      ).addTo(customRouteLayer);

      if (defect.pathCoords) defect.pathCoords.forEach((c) => latlngs.push(c));
      else latlngs.push([defect.lat, defect.lon]);

      const marker = L.marker([defect.lat, defect.lon], {{
        icon: L.divIcon({{
          className: '',
          html: '<div style="width:30px;height:30px;border-radius:15px;background:' + lineColor +
                ';color:white;text-align:center;line-height:30px;font-weight:700;' +
                'border:2px solid white;box-shadow:0 4px 10px rgba(0,0,0,.25);">' + (index + 1) + '</div>',
          iconSize: [30, 30],
          iconAnchor: [15, 15],
        }})
      }});
      const delayPct = Math.round((defect.trafficFactor - 1) * 100);
      const incNote = defect.incidentDelay
        ? '<br><span style="color:#d32f2f;font-weight:700;">\u26a0 Incident congestion on this leg</span>'
        : '';
      marker.bindPopup(
        '<b>Stop ' + (index + 1) + '</b><br>' +
        defect.defect_number + '<br>' +
        defect.specific_location + '<br>' +
        'Arrive: ~' + defect.arrivalTime + '<br>' +
        'Drive: ' + Math.round(defect.driveMin) + ' min' +
          (delayPct > 0 ? ' (+' + delayPct + '% ' + defect.trafficLabel + ' traffic)' : '') +
        incNote + '<br>' +
        'Leg distance: ' + defect.legKm.toFixed(2) + ' km<br>' +
        'Priority score: ' + (defect.priorityScore * 100).toFixed(1) + ' / 100'
      );
      marker.addTo(customRouteLayer);
    }});

    if (latlngs.length > 1) map.fitBounds(latlngs, {{ padding: [50, 50] }});
  }}

  function updateCustomRouteSummary(result) {{
    const summary = document.getElementById('custom-route-summary');
    if (!result.route.length) {{
      summary.innerHTML = 'Choose at least 2 defects to generate a weighted route.';
      return;
    }}
    const depInput = document.getElementById('departure-time-input');
    const depStr = depInput ? depInput.value : hourToLabel({SIMULATION_START_HOUR});
    const rows = result.route.map((d, i) =>
      '<div style="margin-top:5px;"><b>' + (i + 1) + '.</b> ' + d.defect_number +
      ' | ' + d.legKm.toFixed(2) + ' km' +
      ' | drive ' + Math.round(d.driveMin) + ' min' +
      ' | arrive ~' + d.arrivalTime +
      ' | ' + d.trafficLabel + ' traffic' +
      ' | score ' + (d.priorityScore * 100).toFixed(1) + '/100</div>'
    ).join('');
    summary.innerHTML =
      '<b>Route — depart ' + depStr + '</b><br>' +
      '<span class="helper-text">Stops: ' + result.route.length +
      ' | Distance: ' + result.totalKm.toFixed(2) + ' km' +
      ' | Total time: ' + result.totalHours + 'h ' + result.totalMins + 'min (incl. service)</span>' +
      rows;
  }}

  function syncWeightLabels() {{
    [['weight-severity','weight-severity-value'],['weight-road','weight-road-value'],
     ['weight-urgency','weight-urgency-value'],['weight-traffic','weight-traffic-value'],
     ['weight-efficiency','weight-efficiency-value']].forEach(([iId, lId]) => {{
      const inp = document.getElementById(iId);
      const lbl = document.getElementById(lId);
      if (inp && lbl) lbl.textContent = inp.value + '%';
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

  // ── Bootstrap ─────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {{
    syncWeightLabels();
    document.querySelectorAll('.weight-control input[type=range]').forEach((inp) => {{
      inp.addEventListener('input', syncWeightLabels);
    }});

    const generateBtn = document.getElementById('generate-custom-route');
    if (generateBtn) generateBtn.addEventListener('click', generateCustomRoute);

    const selectTopBtn = document.getElementById('select-top-defects');
    if (selectTopBtn) {{
      selectTopBtn.addEventListener('click', () => {{
        Array.from(document.querySelectorAll('.existing-defect-checkbox'))
          .forEach((box, i) => {{ box.checked = i < 10; }});
        generateCustomRoute();
      }});
    }}

    const depInput = document.getElementById('departure-time-input');
    if (depInput) {{
      depInput.addEventListener('change', () => {{
        renderIncidentLayer();
        generateCustomRoute();
      }});
    }}

    // Incident count slider — update label live, apply on release
    const incSlider = document.getElementById('incident-count-slider');
    if (incSlider) {{
      incSlider.addEventListener('input', () => {{
        document.getElementById('incident-count-value').textContent = incSlider.value;
      }});
      incSlider.addEventListener('change', rerandomizeIncidents);
    }}

    setTimeout(() => {{
      // Default: activate 2 incidents on load
      activateIncidentsRandom(2);
      document.getElementById('incident-count-value').textContent = '2';
      rerandomizeIncidents();
    }}, 650);
  }});
</script>
"""

m.get_root().html.add_child(folium.Element(legend_html))

output = "charmhaven_route.html"
m.save(output)
print(f"Map saved: {output}")
