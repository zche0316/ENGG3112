import geopandas as gpd


gdf = gpd.read_file("Dataset\Road_Pothole.kmz")


print(gdf.head())


gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y


clean_df = gdf[["Name", "description", "lat", "lon"]]

clean_df.to_csv("Dataset/potholes_clean.csv", index=False, encoding="utf-8-sig")

print("potholes_clean.csv")