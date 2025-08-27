#%% Import Libraries
import gzip, json
import pandas as pd
from tabulate import tabulate
import re

def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for line in g:
         yield json.loads(line)



#%% Main

# 1) Load
reviews_data = pd.read_json('review_South_Dakota.json.gz', lines=True, compression='gzip')
biz_meta = pd.read_json('meta_South_Dakota.json.gz', lines=True, compression='gzip')

# standardize columns
biz_meta.columns = biz_meta.columns.str.lower().str.strip()
reviews_data.columns = reviews_data.columns.str.lower().str.strip()

#get info of all data
for col in biz_meta.columns:
    print(biz_meta[col].value_counts())

print("\n" + tabulate(reviews_data.head(10), headers="keys", tablefmt="psql"))
print("\n" + tabulate(biz_meta.head(10), headers="keys", tablefmt="psql"))


#%% Data Cleaning
# 1. cleaning of review data
# these columns are IMPT
reviews_data = reviews_data.dropna(subset=["rating", "time", "gmap_id", "user_id"])

# Presence-only (True if not null, False if null)
reviews_data["pics"] = reviews_data["pics"].notna()
reviews_data = reviews_data.rename(columns={"name": "user_name"})



# 2. cleaning of biz meta data
biz_meta = biz_meta.dropna(subset=["gmap_id"])

# text_cols = ["name", "address", "description", "MISC"]
# for col in text_cols:
#     if col in biz_meta.columns:
#         biz_meta[col] = biz_meta[col].apply(clean_text)

# Convert $ → 1, $$ → 2, etc.
biz_meta["price_level"] = biz_meta["price"].str.len()
# Fill missing with 0 = unknown
biz_meta["price_level"] = biz_meta["price_level"].fillna(0).astype("int8")
biz_meta = biz_meta.rename(columns={"name": "biz_name"})

#%% Data Merging
# merge relevant cols from meta to reviews
keep_cols = [
    "gmap_id",        # join key
    "biz_name",
    "description",
    "category",
    "avg_rating",
    "num_of_reviews",
    "hours",
    "address",
    "MISC",
    "price_level",
    "state"
]

keep_cols = [c for c in keep_cols if c in biz_meta.columns]
biz_meta = biz_meta[keep_cols].drop_duplicates(subset=["gmap_id"])

merged_reviews_data = reviews_data.merge(biz_meta, on="gmap_id", how="left")
print("\n" + tabulate(merged_reviews_data.head(10), headers="keys", tablefmt="psql"))

# %%
