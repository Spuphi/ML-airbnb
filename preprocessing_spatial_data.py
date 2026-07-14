import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# CSV laden
listings = pd.read_csv(
    "data/listings_clean.csv",
    usecols=["latitude", "longitude", "price"]
)

print(listings.shape)
print(listings.head())

# GeoJSON laden
districts = gpd.read_file("data/neighbourhoods.geojson")[
    ["neighbourhood", "neighbourhood_group", "geometry"]
]

print(districts.shape)
print(districts.head())

# Listings → GeoDataFrame
listings_gdf = gpd.GeoDataFrame(
    listings,
    geometry=[Point(xy) for xy in zip(listings["longitude"], listings["latitude"])],
    crs="EPSG:4326",
)

# CRS check
assert listings_gdf.crs == districts.crs

# Spatial Join
joined = gpd.sjoin(listings_gdf, districts, how="left", predicate="within")
joined = joined.drop(columns=["index_right", "geometry"])

print("Listings without neighbourhood:", joined["neighbourhood"].isna().sum())
print(joined.head())

# Ergebnis zusammenstellen
result = joined[
    ["latitude", "longitude", "neighbourhood_group", "neighbourhood", "price"]
].rename(columns={
    "neighbourhood_group": "district_group",
    "neighbourhood": "district",
})

# Speichern
result.to_csv("data/listings_spatial.csv", index=False)
print(result.shape)
print(result.head())
