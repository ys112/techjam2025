# %% Import Libraries
import gzip, json
import pandas as pd
from tabulate import tabulate


def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for line in g:
            yield json.loads(line)


# %% Main
# to print
# for i, obj in enumerate(parse("review_South_Dakota.json.gz")):
#     print(obj)
#     if i == 50:
#         break

#! 1) Load
reviews_data = pd.read_json(
    "review_South_Dakota.json.gz", lines=True, compression="gzip"
)  # or .json/.parquet
biz_meta = pd.read_json("meta_South_Dakota.json.gz", lines=True, compression="gzip")

# standardize columns
biz_meta.columns = biz_meta.columns.str.lower().str.strip()
reviews_data.columns = reviews_data.columns.str.lower().str.strip()

print("\n" + tabulate(reviews_data.head(10), headers="keys", tablefmt="psql"))
print("\n" + tabulate(biz_meta.head(10), headers="keys", tablefmt="psql"))

# 1. cleaning of review data
# these columns are NOT NULL
reviews_data = reviews_data.dropna(subset=["text", "rating", "time", "gmap_id", "resp"])

# pics/resp to booleans
reviews_data["has_pics"] = reviews_data["pics"].notna()


# 2. cleaning of meta data
biz_meta = biz_meta.dropna(subset=["gmap_id"])

# Convert $ → 1, $$ → 2, etc.
biz_meta["price_level"] = biz_meta["price"].str.len()
# Fill missing with 0 = unknown
biz_meta["price_level"] = biz_meta["price_level"].fillna(0).astype("int8")


# merge relevant cols from meta to reviews
keep_cols = [
    "gmap_id",  # join key
    "name",  # business name
    "category",  # type of business
    "avg_rating",  # business-level avg
    "num_of_reviews",  # business popularity
    "latitude",
    "longitude",  # optional
    "state",  # active/closed
]

keep_cols = [c for c in keep_cols if c in biz_meta.columns]
meta_small = biz_meta[keep_cols].drop_duplicates(subset=["gmap_id"]).copy()
# %%
