import osmnx as ox
import pandas as pd
import folium


G = ox.load_graphml("Dataset\central_coast_roads.graphml")
df = pd.read_csv("Dataset\potholes_clean.csv")


center_lat = df["lat"].mean()
center_lon = df["lon"].mean()


m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)


edges = ox.graph_to_gdfs(G, nodes=False)


for _, row in edges.iterrows():
    highway = row.get("highway", "")

    if isinstance(highway, list):
        highway = highway[0]

    if highway == "motorway":
        color = "red"
        weight = 3
    elif highway == "primary":
        color = "orange"
        weight = 2.5
    elif highway == "secondary":
        color = "yellow"
        weight = 2
    else:
        color = "gray"
        weight = 1.5

    geom = row.geometry

    if geom.geom_type == "LineString":
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.7
        ).add_to(m)

    elif geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            coords = [(y, x) for x, y in line.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.7
            ).add_to(m)


for _, row in df.iterrows():
    popup_text = f"""
    <b>Name:</b> {row.get('Name', 'N/A')}<br>
    <b>Description:</b> {row.get('description', 'N/A')}<br>
    <b>Latitude:</b> {row['lat']}<br>
    <b>Longitude:</b> {row['lon']}
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.8,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)


legend_html = """
<div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    width: 230px;
    z-index: 9999;
    font-size: 14px;
    background-color: white;
    border: 2px solid grey;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
    <div style="font-weight: bold; margin-bottom: 8px;">Legend</div>

    <div><span style="color:red;">&#8212;&#8212;&#8212;</span> Motorway</div>
    <div><span style="color:orange;">&#8212;&#8212;&#8212;</span> Primary Road</div>
    <div><span style="color:gold;">&#8212;&#8212;&#8212;</span> Secondary Road</div>
    <div><span style="color:gray;">&#8212;&#8212;&#8212;</span> Other Roads</div>
    <div><span style="color:blue;">&#9679;</span> Pothole</div>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

output_file = "NSW_pothole_map.html"
m.save(output_file)

print(f"{output_file}")