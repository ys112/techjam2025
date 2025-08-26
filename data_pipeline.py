#%% Import Libraries
import gzip, json
import pandas as pd
from tabulate import tabulate
import re

def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for line in g:
         yield json.loads(line)

# cleaning all text data columns

# def clean_text(text):
#     if text is None:
#         return pd.NA
#     s = str(text)
#     # collapse whitespace early
#     s = re.sub(r"\s+", " ", s).strip()

#     # normalize spaces around punctuation we keep
#     s = re.sub(r"\s*,\s*", ", ", s)   # exactly one space after commas
#     s = re.sub(r"\s*/\s*", "/", s)    # no spaces around slashes
#     s = re.sub(r"\s*-\s*", "-", s)    # no spaces around hyphens

#     # compress duplicate commas like ", ,"
#     s = re.sub(r",\s*,+", ", ", s)

#     # lowercase at the end
#     s = s.lower().strip(" ,")

#    return s

#%% Main

# 1) Load
reviews_data = pd.read_json('review_South_Dakota.json.gz', lines=True, compression='gzip')        
biz_meta = pd.read_json('meta_South_Dakota.json.gz', lines=True, compression='gzip')

# standardize columns
biz_meta.columns = biz_meta.columns.str.lower().str.strip()
reviews_data.columns = reviews_data.columns.str.lower().str.strip()

print("\n" + tabulate(reviews_data.head(10), headers="keys", tablefmt="psql"))
print("\n" + tabulate(biz_meta.head(10), headers="keys", tablefmt="psql"))


#%% Data Cleaning
# 1. cleaning of review data
# these columns are IMPT
reviews_data = reviews_data.dropna(subset=["rating", "time", "gmap_id", "user_id"])

# text_cols = ["text"]
# for col in text_cols:
#     if col in reviews_data.columns:
#         reviews_data[col] = reviews_data[col].apply(clean_text)

# # Presence-only (True if not null, False if null)
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
