#!/usr/bin/env python3
"""
Crash Location Annotation Tool — Charmhaven Zone
=================================================
Run:   python mark_crashes.py
Then:  A browser window opens with an interactive map.
       Open the Power BI / iROADS report alongside it.
       Select casualty type (or press keys 1-4), then click the map.
       When done, click  Export JSON  → file saves as  crash_data.json
       Tell Claude to read crash_data.json and update ACCIDENT_HOTSPOTS.

Keyboard shortcuts (while map is focused):
  1 = Killed        2 = Seriously Injured
  3 = Moderately    4 = Minor/Other Injured
  Ctrl+Z = undo last marker
"""

import json
import os
import webbrowser

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Current reference hotspots (shown as grey dots on the annotation map) ─────
EXISTING_HOTSPOTS = [
    {"lat": -33.098, "lon": 151.498, "casualty": "Moderately Injured", "road": "Wyee Rd / Pacific Hwy junction"},
    {"lat": -33.104, "lon": 151.558, "casualty": "Moderately Injured", "road": "Pacific Hwy, near Wyee"},
    {"lat": -33.115, "lon": 151.562, "casualty": "Minor/Other Injured","road": "Pacific Hwy, Lake Munmorah N"},
    {"lat": -33.117, "lon": 151.541, "casualty": "Moderately Injured", "road": "Pacific Hwy, Doyalson North"},
    {"lat": -33.121, "lon": 151.562, "casualty": "Moderately Injured", "road": "Pacific Hwy, Lake Munmorah"},
    {"lat": -33.122, "lon": 151.576, "casualty": "Seriously Injured",  "road": "Pacific Hwy, Lake Munmorah E"},
    {"lat": -33.131, "lon": 151.543, "casualty": "Moderately Injured", "road": "Scenic Dr, Doyalson"},
    {"lat": -33.135, "lon": 151.571, "casualty": "Moderately Injured", "road": "Pacific Hwy, Lake Munmorah S"},
    {"lat": -33.147, "lon": 151.537, "casualty": "Moderately Injured", "road": "Scenic Dr, Doyalson S"},
    {"lat": -33.155, "lon": 151.524, "casualty": "Seriously Injured",  "road": "Doyalson Link Rd"},
    {"lat": -33.158, "lon": 151.522, "casualty": "Moderately Injured", "road": "Doyalson Link Rd"},
    {"lat": -33.157, "lon": 151.541, "casualty": "Moderately Injured", "road": "Scenic Dr, Blue Haven N"},
    {"lat": -33.168, "lon": 151.537, "casualty": "Moderately Injured", "road": "Scenic Dr, Blue Haven N"},
    {"lat": -33.183, "lon": 151.537, "casualty": "Minor/Other Injured","road": "Scenic Dr, Colongra N"},
    {"lat": -33.193, "lon": 151.537, "casualty": "Seriously Injured",  "road": "Colongra / Scenic Dr junction"},
    {"lat": -33.206, "lon": 151.498, "casualty": "Minor/Other Injured","road": "Pacific Hwy, San Remo N"},
    {"lat": -33.213, "lon": 151.499, "casualty": "Moderately Injured", "road": "Pacific Hwy, San Remo"},
    {"lat": -33.221, "lon": 151.503, "casualty": "Seriously Injured",  "road": "Pacific Hwy, Blue Haven / San Remo junction"},
    {"lat": -33.228, "lon": 151.499, "casualty": "Seriously Injured",  "road": "Pacific Hwy, Blue Haven"},
    {"lat": -33.234, "lon": 151.496, "casualty": "Minor/Other Injured","road": "Pacific Hwy, Charmhaven N"},
    {"lat": -33.241, "lon": 151.494, "casualty": "Moderately Injured", "road": "Pacific Hwy, Charmhaven"},
    {"lat": -33.248, "lon": 151.497, "casualty": "Seriously Injured",  "road": "Pacific Hwy, Charmhaven mid"},
    {"lat": -33.253, "lon": 151.499, "casualty": "Killed",             "road": "Pacific Hwy, Charmhaven"},
    {"lat": -33.260, "lon": 151.501, "casualty": "Minor/Other Injured","road": "Pacific Hwy, Charmhaven S"},
    {"lat": -33.265, "lon": 151.508, "casualty": "Moderately Injured", "road": "Pacific Hwy, near Wyong Public Hospital"},
    {"lat": -33.268, "lon": 151.510, "casualty": "Seriously Injured",  "road": "Pacific Hwy / Hamlyn Terrace junction"},
    {"lat": -33.278, "lon": 151.511, "casualty": "Moderately Injured", "road": "Pacific Hwy, Kanwal"},
    {"lat": -33.282, "lon": 151.510, "casualty": "Minor/Other Injured","road": "Pacific Hwy, Kanwal S"},
    {"lat": -33.162, "lon": 151.437, "casualty": "Moderately Injured", "road": "Pacific Motorway (M1)"},
    {"lat": -33.228, "lon": 151.430, "casualty": "Moderately Injured", "road": "Pacific Motorway (M1), Halloran N"},
    {"lat": -33.235, "lon": 151.435, "casualty": "Moderately Injured", "road": "Pacific Motorway (M1), Halloran"},
    {"lat": -33.244, "lon": 151.442, "casualty": "Seriously Injured",  "road": "Pacific Motorway (M1) / Halloran junction"},
    {"lat": -33.255, "lon": 151.447, "casualty": "Seriously Injured",  "road": "Pacific Motorway (M1) / B70 junction"},
    {"lat": -33.255, "lon": 151.410, "casualty": "Seriously Injured",  "road": "Jilliby Road"},
    {"lat": -33.258, "lon": 151.494, "casualty": "Moderately Injured", "road": "Minnesota Rd, Hamlyn Terrace"},
    {"lat": -33.258, "lon": 151.504, "casualty": "Minor/Other Injured","road": "Near Wyong Public Hospital entrance"},
    {"lat": -33.263, "lon": 151.497, "casualty": "Seriously Injured",  "road": "Hamlyn Terrace"},
    {"lat": -33.266, "lon": 151.493, "casualty": "Moderately Injured", "road": "Hamlyn Terrace"},
    {"lat": -33.267, "lon": 151.479, "casualty": "Seriously Injured",  "road": "B70 W, Hamlyn Terrace"},
    {"lat": -33.270, "lon": 151.488, "casualty": "Seriously Injured",  "road": "B70, Hamlyn Terrace mid"},
    {"lat": -33.272, "lon": 151.480, "casualty": "Moderately Injured", "road": "Louisiana Rd / B70, Hamlyn Terrace"},
    {"lat": -33.271, "lon": 151.476, "casualty": "Seriously Injured",  "road": "B70 junction W"},
    {"lat": -33.276, "lon": 151.470, "casualty": "Moderately Injured", "road": "B70 W"},
    {"lat": -33.277, "lon": 151.483, "casualty": "Minor/Other Injured","road": "B70 mid"},
    {"lat": -33.265, "lon": 151.498, "casualty": "Moderately Injured", "road": "B70 E, Hamlyn Terrace"},
    {"lat": -33.272, "lon": 151.512, "casualty": "Moderately Injured", "road": "B70, near Gorokan"},
    {"lat": -33.273, "lon": 151.487, "casualty": "Moderately Injured", "road": "Wadalba Road"},
    {"lat": -33.275, "lon": 151.491, "casualty": "Minor/Other Injured","road": "Wadalba Road"},
    {"lat": -33.279, "lon": 151.451, "casualty": "Moderately Injured", "road": "Pacific Hwy W, Wadalba"},
    {"lat": -33.280, "lon": 151.455, "casualty": "Seriously Injured",  "road": "Pacific Hwy junction, Wadalba"},
    {"lat": -33.285, "lon": 151.457, "casualty": "Moderately Injured", "road": "Pacific Hwy S/W"},
    {"lat": -33.282, "lon": 151.444, "casualty": "Seriously Injured",  "road": "Alison / Wadalba junction"},
    {"lat": -33.279, "lon": 151.478, "casualty": "Seriously Injured",  "road": "Wadalba main road"},
    {"lat": -33.294, "lon": 151.476, "casualty": "Moderately Injured", "road": "Paclock Ave, Wyong"},
    {"lat": -33.292, "lon": 151.462, "casualty": "Seriously Injured",  "road": "Wyong area"},
    {"lat": -33.294, "lon": 151.431, "casualty": "Moderately Injured", "road": "Wyong North"},
    {"lat": -33.298, "lon": 151.428, "casualty": "Moderately Injured", "road": "Wyong North"},
    {"lat": -33.297, "lon": 151.444, "casualty": "Seriously Injured",  "road": "Wyong"},
    {"lat": -33.265, "lon": 151.398, "casualty": "Moderately Injured", "road": "Jilliby area road"},
    {"lat": -33.266, "lon": 151.426, "casualty": "Moderately Injured", "road": "Halloran area road"},
    {"lat": -33.274, "lon": 151.429, "casualty": "Moderately Injured", "road": "Alison Road"},
    {"lat": -33.277, "lon": 151.443, "casualty": "Seriously Injured",  "road": "Alison Road"},
    {"lat": -33.281, "lon": 151.458, "casualty": "Moderately Injured", "road": "Alison Road S"},
    {"lat": -33.256, "lon": 151.524, "casualty": "Moderately Injured", "road": "Gorokan N"},
    {"lat": -33.260, "lon": 151.536, "casualty": "Minor/Other Injured","road": "Gorokan"},
    {"lat": -33.265, "lon": 151.525, "casualty": "Moderately Injured", "road": "Marks Rd, Gorokan"},
    {"lat": -33.265, "lon": 151.533, "casualty": "Moderately Injured", "road": "Gorokan main road"},
    {"lat": -33.267, "lon": 151.549, "casualty": "Seriously Injured",  "road": "Budgewoi Rd / Toukley W"},
    {"lat": -33.268, "lon": 151.551, "casualty": "Moderately Injured", "road": "Toukley B70"},
    {"lat": -33.269, "lon": 151.553, "casualty": "Seriously Injured",  "road": "Toukley"},
    {"lat": -33.268, "lon": 151.558, "casualty": "Moderately Injured", "road": "Toukley / Noraville"},
    {"lat": -33.271, "lon": 151.554, "casualty": "Moderately Injured", "road": "Noraville N"},
    {"lat": -33.274, "lon": 151.561, "casualty": "Moderately Injured", "road": "Noraville"},
    {"lat": -33.275, "lon": 151.565, "casualty": "Seriously Injured",  "road": "Noraville"},
    {"lat": -33.294, "lon": 151.558, "casualty": "Seriously Injured",  "road": "Rossetti Lookout N"},
    {"lat": -33.297, "lon": 151.564, "casualty": "Moderately Injured", "road": "Rossetti Lookout"},
    {"lat": -33.229, "lon": 151.573, "casualty": "Moderately Injured", "road": "Budgewoi N"},
    {"lat": -33.234, "lon": 151.565, "casualty": "Seriously Injured",  "road": "Buff Point"},
    {"lat": -33.237, "lon": 151.562, "casualty": "Moderately Injured", "road": "Buff Point"},
    {"lat": -33.238, "lon": 151.567, "casualty": "Seriously Injured",  "road": "Buff Point / A9"},
    {"lat": -33.243, "lon": 151.572, "casualty": "Seriously Injured",  "road": "Halekulani N"},
    {"lat": -33.235, "lon": 151.567, "casualty": "Seriously Injured",  "road": "Halekulani"},
    {"lat": -33.238, "lon": 151.573, "casualty": "Moderately Injured", "road": "Halekulani"},
    {"lat": -33.252, "lon": 151.579, "casualty": "Seriously Injured",  "road": "Halekulani S"},
    {"lat": -33.259, "lon": 151.571, "casualty": "Moderately Injured", "road": "Halekulani S / Budgewoi"},
    {"lat": -33.274, "lon": 151.580, "casualty": "Moderately Injured", "road": "Elizabeth Bay Dr"},
    {"lat": -33.285, "lon": 151.570, "casualty": "Seriously Injured",  "road": "Elizabeth Bay Dr S"},
    {"lat": -33.293, "lon": 151.573, "casualty": "Moderately Injured", "road": "Noraville S"},
]

existing_json = json.dumps(EXISTING_HOTSPOTS, ensure_ascii=False)

# ── HTML annotation tool ───────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crash Annotation — Charmhaven Zone 2024</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{display:flex;font-family:"Segoe UI","Helvetica Neue",sans-serif;font-size:13px;color:#1f2937;}
#map{flex:1;}
#panel{width:330px;min-width:330px;background:#fff;border-left:1px solid #dde3ed;
       display:flex;flex-direction:column;overflow:hidden;box-shadow:-4px 0 12px rgba(0,0,0,.07);}
#panel-head{padding:14px 16px 12px;border-bottom:1px solid #e5e7eb;flex-shrink:0;}
#panel-scroll{overflow-y:auto;flex:1;padding:14px 16px;display:flex;flex-direction:column;gap:13px;}
h2{font-size:12px;font-weight:700;color:#374151;margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em;}
.info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
      padding:10px 12px;font-size:11px;line-height:1.55;color:#1e40af;}
.kbd{display:inline-block;background:#e5e7eb;border:1px solid #d1d5db;border-radius:4px;
     padding:1px 5px;font-family:monospace;font-size:10px;}

/* Casualty buttons */
.cbt{display:flex;align-items:center;gap:9px;padding:9px 12px;border:2px solid #e5e7eb;
     border-radius:9px;cursor:pointer;font:inherit;font-size:12px;font-weight:600;
     background:#fafafa;color:#374151;width:100%;text-align:left;transition:all .12s;}
.cbt:hover{background:#f1f5f9;border-color:#d1d5db;}
.cbt.active{background:#fff;box-shadow:0 0 0 3px currentColor33;}
.cbt.active[data-c="Killed"]{border-color:#e53935;background:#fff5f5;}
.cbt.active[data-c="Seriously Injured"]{border-color:#fb8c00;background:#fff8f0;}
.cbt.active[data-c="Moderately Injured"]{border-color:#37474f;background:#f4f6f6;}
.cbt.active[data-c="Minor/Other Injured"]{border-color:#00acc1;background:#f0fdff;}
.cdot{width:13px;height:13px;border-radius:50%;flex-shrink:0;border:1.5px solid rgba(0,0,0,.15);}

/* Road input */
.rinput{width:100%;border:1px solid #d1d5db;border-radius:8px;padding:9px 11px;
        font:inherit;font-size:12px;background:#fafafa;}
.rinput:focus{outline:none;border-color:#3b82f6;background:#fff;box-shadow:0 0 0 3px #bfdbfe55;}

/* Stats */
.sgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;}
.scard{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
       padding:7px 9px;text-align:center;}
.slabel{font-size:10px;color:#6b7280;font-weight:600;}
.sval{font-size:20px;font-weight:800;line-height:1.2;}

/* Buttons */
.btn{border:none;border-radius:8px;padding:9px 13px;font:inherit;font-weight:600;
     cursor:pointer;font-size:12px;transition:background .12s;}
.bp{background:#1d4ed8;color:#fff;} .bp:hover{background:#1e40af;}
.bs{background:#f1f5f9;color:#374151;} .bs:hover{background:#e2e8f0;}
.bd{background:#fee2e2;color:#dc2626;} .bd:hover{background:#fecaca;}
.brow{display:flex;gap:6px;}

/* Marker list */
.mlist{max-height:180px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:8px;}
.mitem{display:flex;align-items:center;gap:7px;padding:6px 9px;
       border-bottom:1px solid #f3f4f6;font-size:11px;}
.mitem:last-child{border-bottom:none;}
.mitem:nth-child(even){background:#fafafa;}
.mrm{margin-left:auto;background:none;border:none;cursor:pointer;
     color:#9ca3af;font-size:15px;padding:0 3px;line-height:1;}
.mrm:hover{color:#dc2626;}

/* JSON output */
#jout{width:100%;height:90px;border:1px solid #d1d5db;border-radius:8px;
      padding:8px;font-size:10px;font-family:monospace;resize:vertical;color:#374151;}

/* Toggle */
.trow{display:flex;align-items:center;justify-content:space-between;font-size:12px;}
.tsw{width:36px;height:20px;background:#d1d5db;border-radius:10px;position:relative;
     cursor:pointer;transition:background .2s;flex-shrink:0;}
.tsw.on{background:#3b82f6;}
.tsw::after{content:'';position:absolute;width:16px;height:16px;background:#fff;
            border-radius:50%;top:2px;left:2px;transition:transform .2s;
            box-shadow:0 1px 3px rgba(0,0,0,.3);}
.tsw.on::after{transform:translateX(16px);}

/* Coord bar */
#cbar{font-size:10px;color:#6b7280;font-family:monospace;text-align:center;
      padding:4px 0;background:#f8fafc;border-radius:6px;border:1px solid #e5e7eb;}

.section-sep{border:none;border-top:1px solid #f3f4f6;margin:2px 0;}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <div id="panel-head">
    <div style="font-size:16px;font-weight:800;">Crash Annotation</div>
    <div style="font-size:11px;color:#6b7280;margin-top:2px;">
      Charmhaven Zone — NSW 2024 iROADS data
    </div>
  </div>
  <div id="panel-scroll">

    <div class="info">
      <b>How to use:</b><br>
      1. Open Power BI report alongside this window<br>
      2. Select casualty type (or press <span class="kbd">1</span>–<span class="kbd">4</span>)<br>
      3. Type road name, then <b>click the map</b><br>
      4. Click <b>Export JSON</b> when done → send the file to Claude
    </div>

    <div>
      <h2>Casualty Type</h2>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <button class="cbt" data-c="Killed" style="color:#e53935" onclick="selectC(this)">
          <div class="cdot" style="background:#e53935;"></div>Killed
          <span style="margin-left:auto;font-size:10px;font-weight:400;color:#9ca3af;"><kbd class="kbd">1</kbd></span>
        </button>
        <button class="cbt" data-c="Seriously Injured" style="color:#f57c00" onclick="selectC(this)">
          <div class="cdot" style="background:#fb8c00;"></div>Seriously Injured
          <span style="margin-left:auto;font-size:10px;font-weight:400;color:#9ca3af;"><kbd class="kbd">2</kbd></span>
        </button>
        <button class="cbt active" data-c="Moderately Injured" style="color:#37474f" onclick="selectC(this)">
          <div class="cdot" style="background:#37474f;"></div>Moderately Injured
          <span style="margin-left:auto;font-size:10px;font-weight:400;color:#9ca3af;"><kbd class="kbd">3</kbd></span>
        </button>
        <button class="cbt" data-c="Minor/Other Injured" style="color:#00838f" onclick="selectC(this)">
          <div class="cdot" style="background:#00acc1;"></div>Minor / Other Injured
          <span style="margin-left:auto;font-size:10px;font-weight:400;color:#9ca3af;"><kbd class="kbd">4</kbd></span>
        </button>
      </div>
    </div>

    <div>
      <h2>Road Name <span style="font-weight:400;color:#9ca3af;">(optional)</span></h2>
      <input class="rinput" type="text" id="road" placeholder="e.g. Pacific Hwy, Charmhaven" list="rsugg">
      <datalist id="rsugg">
        <option value="Pacific Highway (A43)">
        <option value="Pacific Motorway (M1)">
        <option value="B70, Hamlyn Terrace">
        <option value="Budgewoi Road">
        <option value="Scenic Drive">
        <option value="Doyalson Link Road">
        <option value="Minnesota Road, Hamlyn Terrace">
        <option value="Louisiana Road, Hamlyn Terrace">
        <option value="Marks Road, Gorokan">
        <option value="Alison Road">
        <option value="Elizabeth Bay Drive">
        <option value="Wyong Road">
        <option value="Main Road, Toukley">
        <option value="Wallarah Road, Gorokan">
      </datalist>
    </div>

    <div>
      <h2>Progress</h2>
      <div class="sgrid">
        <div class="scard" style="grid-column:span 3;">
          <div class="slabel">Total Marked</div>
          <div class="sval" id="ct">0</div>
        </div>
        <div class="scard">
          <div class="slabel" style="color:#e53935;">Killed</div>
          <div class="sval" id="ck" style="color:#e53935;">0</div>
        </div>
        <div class="scard">
          <div class="slabel" style="color:#fb8c00;">Seriously</div>
          <div class="sval" id="cs" style="color:#fb8c00;">0</div>
        </div>
        <div class="scard">
          <div class="slabel" style="color:#37474f;">Moderately</div>
          <div class="sval" id="cm" style="color:#37474f;">0</div>
        </div>
        <div class="scard" style="grid-column:span 3;">
          <div class="slabel" style="color:#00838f;">Minor / Other</div>
          <div class="sval" id="cn" style="color:#00acc1;">0</div>
        </div>
      </div>
      <div id="cbar" style="margin-top:6px;">Click map to place markers</div>
    </div>

    <div>
      <h2>Recent Markers <span style="font-weight:400;color:#9ca3af;">× to remove</span></h2>
      <div class="mlist" id="mlist"><div style="padding:10px;color:#9ca3af;font-size:11px;text-align:center;">No markers yet</div></div>
    </div>

    <div>
      <h2>Actions</h2>
      <div class="brow">
        <button class="btn bp" onclick="exportJSON()" style="flex:1;">⬇ Export JSON</button>
        <button class="btn bs" onclick="copyJSON()">⎘ Copy</button>
      </div>
      <div class="brow" style="margin-top:6px;">
        <button class="btn bs" onclick="undoLast()" style="flex:1;">↩ Undo Last</button>
        <button class="btn bd" onclick="clearAll()">✕ Clear All</button>
      </div>
    </div>

    <div>
      <div class="trow">
        <span>Show existing reference markers</span>
        <div class="tsw on" id="reftog" onclick="toggleRef(this)"></div>
      </div>
      <div style="font-size:10px;color:#9ca3af;margin-top:3px;">
        Grey dots = current ACCIDENT_HOTSPOTS (88 estimated locations)
      </div>
    </div>

    <div>
      <h2>JSON Output <span style="font-weight:400;color:#9ca3af;">— paste to Claude if download fails</span></h2>
      <textarea id="jout" readonly placeholder='[]'></textarea>
    </div>

  </div>
</div>

<script>
const COLORS = {
  'Killed':             '#e53935',
  'Seriously Injured':  '#fb8c00',
  'Moderately Injured': '#37474f',
  'Minor/Other Injured':'#00acc1',
};
const SIZE = {
  'Killed': 15,
  'Seriously Injured': 13,
  'Moderately Injured': 10,
  'Minor/Other Injured': 9,
};

const existingHotspots = EXISTING_JSON_PLACEHOLDER;

let selected = 'Moderately Injured';
let placed = [];       // {lat,lon,casualty,road,marker}
let refLayer = null;
let showRef = true;

// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map('map', {zoomControl:true}).setView([-33.22, 151.505], 14);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors', maxZoom: 19
}).addTo(map);

// Satellite option
const esri = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Esri World Imagery', maxZoom:19}
);
L.control.layers({'Street (OSM)': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {attribution:'© OSM',maxZoom:19}).addTo(map), 'Satellite (Esri)': esri}).addTo(map);

// Reference layer
refLayer = L.layerGroup();
existingHotspots.forEach(h => {
  L.circleMarker([h.lat, h.lon], {
    radius: 5, color:'#9ca3af', fillColor:'#9ca3af',
    fillOpacity:0.45, weight:1.5
  }).bindPopup('<span style="color:#6b7280;font-size:12px;">[Existing estimate]</span><br><b>' +
    h.casualty + '</b><br>' + h.road).addTo(refLayer);
});
refLayer.addTo(map);

// Click to place marker
map.on('click', e => {
  const lat = +e.latlng.lat.toFixed(5);
  const lon = +e.latlng.lng.toFixed(5);
  const road = document.getElementById('road').value.trim() || 'Road not specified';
  placeMarker(lat, lon, selected, road);
});

// ── Marker functions ──────────────────────────────────────────────────────────
function makeIcon(casualty) {
  const c = COLORS[casualty] || '#888';
  const s = SIZE[casualty] || 10;
  return L.divIcon({
    className:'',
    html:'<div style="width:'+s+'px;height:'+s+'px;border-radius:50%;background:'+c+
         ';border:2px solid white;box-shadow:0 1px 5px rgba(0,0,0,.45);"></div>',
    iconSize:[s,s], iconAnchor:[s/2,s/2]
  });
}

function placeMarker(lat, lon, casualty, road) {
  const m = L.marker([lat,lon], {icon: makeIcon(casualty)});
  m.bindPopup(
    '<b>' + casualty + '</b><br>' + road +
    '<br><small style="color:#6b7280;">' + lat + ', ' + lon + '</small><br>' +
    '<button onclick="rmByLL('+lat+','+lon+')" ' +
    'style="margin-top:6px;font-size:11px;background:#fee2e2;border:none;' +
    'border-radius:4px;padding:3px 8px;cursor:pointer;color:#dc2626;">Remove</button>'
  );
  m.addTo(map);
  placed.push({lat, lon, casualty, road, marker: m});
  document.getElementById('cbar').textContent =
    'Last: '+lat+', '+lon+'  —  '+casualty;
  updateUI();
}

function rmByLL(lat, lon) {
  const i = placed.findIndex(p => p.lat===lat && p.lon===lon);
  if (i !== -1) { map.removeLayer(placed[i].marker); placed.splice(i,1); updateUI(); }
  map.closePopup();
}

function undoLast() {
  if (!placed.length) return;
  const last = placed.pop();
  map.removeLayer(last.marker);
  updateUI();
}

function clearAll() {
  if (!placed.length) return;
  if (!confirm('Remove all ' + placed.length + ' placed markers?')) return;
  placed.forEach(p => map.removeLayer(p.marker));
  placed = [];
  updateUI();
}

function toggleRef(el) {
  showRef = !showRef;
  el.classList.toggle('on', showRef);
  showRef ? refLayer.addTo(map) : map.removeLayer(refLayer);
}

// ── UI update ─────────────────────────────────────────────────────────────────
function updateUI() {
  const cnts = {};
  Object.keys(COLORS).forEach(k => cnts[k] = 0);
  placed.forEach(p => { cnts[p.casualty] = (cnts[p.casualty]||0)+1; });
  document.getElementById('ct').textContent = placed.length;
  document.getElementById('ck').textContent = cnts['Killed'];
  document.getElementById('cs').textContent = cnts['Seriously Injured'];
  document.getElementById('cm').textContent = cnts['Moderately Injured'];
  document.getElementById('cn').textContent = cnts['Minor/Other Injured'];

  const list = document.getElementById('mlist');
  if (!placed.length) {
    list.innerHTML = '<div style="padding:10px;color:#9ca3af;font-size:11px;text-align:center;">No markers yet</div>';
  } else {
    list.innerHTML = placed.slice().reverse().slice(0,30).map(p => {
      const c = COLORS[p.casualty]||'#888';
      return '<div class="mitem">' +
        '<div style="width:9px;height:9px;border-radius:50%;background:'+c+';flex-shrink:0;"></div>' +
        '<div style="flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">' +
          p.road + '</div>' +
        '<button class="mrm" onclick="rmByLL('+p.lat+','+p.lon+')">×</button>' +
      '</div>';
    }).join('');
  }

  document.getElementById('jout').value = getJSON();
}

function getJSON() {
  return JSON.stringify(
    placed.map(({lat,lon,casualty,road}) => ({lat,lon,casualty,road})),
    null, 2
  );
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportJSON() {
  if (!placed.length) { alert('No markers placed yet.'); return; }
  const blob = new Blob([getJSON()], {type:'application/json'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'crash_data.json'; a.click();
  URL.revokeObjectURL(url);
}

function copyJSON() {
  if (!placed.length) { alert('No markers placed yet.'); return; }
  navigator.clipboard.writeText(getJSON()).then(() => {
    const btn = document.querySelector('.bp + .btn');
    if (btn) { btn.textContent = '✓ Copied!'; setTimeout(()=>btn.textContent='⎘ Copy',2000); }
  });
}

// ── Casualty selector ─────────────────────────────────────────────────────────
function selectC(btn) {
  document.querySelectorAll('.cbt').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selected = btn.dataset.c;
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  const map = {'1':'Killed','2':'Seriously Injured','3':'Moderately Injured','4':'Minor/Other Injured'};
  if (map[e.key]) {
    const btn = document.querySelector('[data-c="'+map[e.key]+'"]');
    if (btn) selectC(btn);
  }
  if (e.ctrlKey && e.key==='z') { e.preventDefault(); undoLast(); }
});
</script>
</body>
</html>
"""

# Inject the existing hotspots JSON into the HTML
HTML = HTML.replace("EXISTING_JSON_PLACEHOLDER", existing_json)

# Write the file
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_marker.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print("=" * 56)
print("  Crash Annotation Tool — Charmhaven Zone 2024")
print("=" * 56)
print(f"\n  Generated: crash_marker.html")
print("\n  Instructions:")
print("  1. Open the Power BI report in one window/tab")
print("  2. Use crash_marker.html in another window")
print("  3. Select casualty type (keys 1-4), type road name,")
print("     then click the map to place a marker")
print("  4. Click  Export JSON  → saves crash_data.json")
print("  5. Tell Claude: 'read crash_data.json and update")
print("     ACCIDENT_HOTSPOTS in Charmhaven_Route_Planner.py'")
print("\n  Grey dots = current estimated positions (for reference)")
print("  Satellite layer available in the map layer control (↗)")
print("=" * 56)

# Auto-open in browser
webbrowser.open("file:///" + out_path.replace("\\", "/"))
