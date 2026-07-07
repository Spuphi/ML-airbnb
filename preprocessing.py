import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

listing_file_path = "data/listings.csv"
listing_data = pd.read_csv("data/listings.csv", low_memory=False) # low_memory liest große Dateien sauberer
print(listing_data.head())

print(listing_data.shape)           # wie viele Zeilen/Spalten?
print(listing_data.columns.tolist()) # welche Spalten gibt es?
print(listing_data["price"].head(10)) # wie sieht der Preis aus?

#-----------------------------------------------------------------
# 1. Preis bereinigen
listing_data["price"] = listing_data["price"].astype(str).str.replace(r"[$,]", "", regex=True)
listing_data["price"] = pd.to_numeric(listing_data["price"], errors="coerce")

# Ausreißer Zeile entfernen
listing_data = listing_data[(listing_data["price"] > 0) & (listing_data["price"] <= 2000)]

# Zeilen ohne Preis löschen
listing_data = listing_data.dropna(subset=["price"])

print(listing_data["price"].head(10))
#-----------------------------------------------------------------
# 2. Relevante Spalten auswählen
keep_columns = ["price",
    "room_type", "property_type", "accommodates", "bathrooms", "bedrooms", "beds",
    "neighbourhood_cleansed", "latitude", "longitude",
    "minimum_nights", "maximum_nights", "instant_bookable",
    "availability_365",
    "number_of_reviews", "review_scores_rating", "review_scores_cleanliness",
    "review_scores_location", "review_scores_value", "reviews_per_month",
    "host_is_superhost", "host_response_rate", "host_total_listings_count",
    "name", "description", "amenities",
]
listing_data = listing_data[keep_columns]
print(listing_data.columns.tolist)

#-----------------------------------------------------------------
#3. Kategorische Spalten finden
s = listing_data.dtypes.astype(str).isin(["object", "str", "string"])
object_cols = list(s[s].index)
print(object_cols)

# Welche davon sind wirklich Kategorisch?
for col in object_cols: # gehe jede Spalte durch, col ist aktueller spaltenname
    n_unique = listing_data[col].nunique() # listing_data[col] greift auf spalte zu und zählt wie viele unterschiedliche werte
    print(f"{col}: {n_unique} eindeutige Werte")

# host_response_rate ist unklar -> anschauen
print(listing_data["host_response_rate"].head(10))


# host_response_rate bereinigen
listing_data["host_response_rate"] = ( # entferne % und überschreibe alte (immer noch String)
    listing_data["host_response_rate"]
    .str.replace("%", "", regex=False)
)
listing_data["host_response_rate"] = pd.to_numeric( # wandelt Text in int um, 
    listing_data["host_response_rate"], errors="coerce" # falls kein Wert umgewandelt werden kann -> Nan
)
# NaN auffüllen mit median
listing_data["host_response_rate"] = listing_data["host_response_rate"].fillna(
    listing_data["host_response_rate"].median()
) 

print(listing_data["host_response_rate"].head(10))

#-----------------------------------------------------------------
# 4. NaN bei kategorischen Spalten behandeln
# Welche Spalten haben NaN
for col in ["room_type", "property_type", "neighbourhood_cleansed", "instant_bookable", "host_is_superhost"]:
    print(f"{col}: {listing_data[col].isna().sum()} fehlende Werte")  # -> host_is_superhost hat fehlende Werte

print("Zeilen: ", listing_data["host_is_superhost"].shape) # 434 von 7008 Zeilen haben NaN
# Fehlende Werte mit Imputation füllen -> höchsten Wert nehmen
imputer = SimpleImputer(strategy="most_frequent")
imputer.fit(listing_data[["host_is_superhost"]])  # lernt welche Wert am häufigsten vorkommt

# gelernten wert anwenden, also NaN füllen
# imputer.transform() gibt ein 2D-Array zurück, aber pandas erwartet 1D für eine einzelne Spalte -> ravel()
listing_data["host_is_superhost"] = imputer.transform(listing_data[["host_is_superhost"]]).ravel() 

# nochmmal testen, # Welche Spalten haben NaN
for col in ["room_type", "property_type", "neighbourhood_cleansed", "instant_bookable", "host_is_superhost"]:
    print(f"{col}: {listing_data[col].isna().sum()} fehlende Werte")

#-----------------------------------------------------------------
# 5. Kategorische Spalten encoden
# Binäre Spalten (instant_bookable, host_is_superhost)
# instant_bookable
print(listing_data["instant_bookable"].head(10)) # werte werden mit t und f gekennzeichnet
listing_data["instant_bookable"] = listing_data["instant_bookable"].map({"t": 1, "f": 0})

# host_is_superhost
print(listing_data["host_is_superhost"].head(10))
listing_data["host_is_superhost"] = listing_data["host_is_superhost"].map({"t": 1, "f": 0})

print(listing_data["instant_bookable"].head(10))
print(listing_data["host_is_superhost"].head(10))
print(listing_data.shape)

# One-Hot:
print(listing_data["room_type"].unique())
# room type -> 4 verschiedene Typen ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room']
OH_encoder = OneHotEncoder(sparse_output=False)
#OneHotEncoder erwartet 2D Daten also DataFrame oder Array, hier nur eine Series! (also Inhalt einer Spalte)
# -> übergebe Spalte als DataFrame mit extra []
OH_room_type = pd.DataFrame(OH_encoder.fit_transform(listing_data[["room_type"]])) # lernt namen und erzeugt spalten

# index zurücksetzen, damit nicht falsch nummeriert sind nach fit_transform
OH_room_type.index = listing_data.index

# Namen zurücksetzen
OH_room_type.columns = OH_encoder.get_feature_names_out(["room_type"])

# entferne kategorische spalten (werden mit one-hot spalten gefüllt) -> haben jetzt numerische Spalten
onehot_cols = ["room_type"]
num_room_type = listing_data.drop(onehot_cols, axis=1)

# füge jetzt one-hot spalten zu numerischen spalten hinzu
clean_data = pd.concat([num_room_type, OH_room_type], axis=1)

# spaltennamen sollen strings sein
clean_data.columns = clean_data.columns.astype(str)

print(clean_data.shape)
# Sollte sein: gleiche Zeilenanzahl wie vorher, aber Spaltenanzahl = 
# (alte Spalten - 1 [room_type weg]) + 4 [neue One-Hot-Spalten] -> also 3 mehr -> passt


# ----------------------------------------------------------------
# Label encoding - ersetzt einfach text durch zahl
le = LabelEncoder()
# property_type
print(clean_data["property_type"].head(10))
# gibt array mit einer Spalte zurück, kann daher direkt in die dataframe-spalte von clean_data geschrieben bzw. ersetzt werden
clean_data["property_type"] = le.fit_transform(clean_data["property_type"]) 
print(clean_data["property_type"].head(10))

# neighbourhood_cleansed

print(clean_data["neighbourhood_cleansed"].head(10))
clean_data["neighbourhood_cleansed"] = le.fit_transform(clean_data["neighbourhood_cleansed"]) 
print(clean_data["neighbourhood_cleansed"].head(10))

print(clean_data.isna().sum().sum())
# Sollte 0 sein! Falls hier eine große Zahl auftaucht, war der Index-Fix nicht korrekt

#-------------------------------------------------------------------------------------
# 5. Restliche numerische Spalten auf NaN prüfen
num_cols = ["bathrooms", "bedrooms", "beds", "review_scores_rating",
            "review_scores_cleanliness", "review_scores_location",
            "review_scores_value", "reviews_per_month", "host_total_listings_count"]

for col in num_cols:
    print(f"{col}: {clean_data[col].isna().sum()} fehlende Werte")

# Spalten mit wenig fehlende Werten -> Zeilen droppen
clean_data = clean_data.dropna(subset=["bathrooms","bedrooms","host_total_listings_count"])

for col in num_cols:
    print(f"{col}: {clean_data[col].isna().sum()} fehlende Werte")

# Spalten mit vielen fehlenden Werten -> median füllen
print(clean_data["review_scores_rating"].head(10))

num_cols = ["review_scores_rating", "review_scores_cleanliness", 
            "review_scores_location","review_scores_value", "reviews_per_month"]

for col in num_cols:
    median_val = clean_data[col].median()
    clean_data[col] = clean_data[col].fillna(median_val)

#------------------------------------------------------------------------
# 6. Freitext spalten NaN auffüllen
for col in ["name", "description", "amenities"]:
    clean_data[col] = clean_data[col].fillna("")

# 7. Dataframe prüfen und speichern
print(clean_data.isna().sum())

clean_data.to_csv("data/listings_clean.csv", index=False)
print("Gespeichert als listings_clean.csv")