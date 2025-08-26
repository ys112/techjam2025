#%% Import Libraries
import gzip, json
import pandas as pd
from tabulate import tabulate

def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for line in g:
         yield json.loads(line)


#%% Main
# to print
# for i, obj in enumerate(parse("review_South_Dakota.json.gz")):
#     print(obj)
#     if i == 50:
#         break

#! 1) Load
reviews_data = pd.read_json('review_South_Dakota.json.gz', lines=True, compression='gzip')         # or .json/.parquet
biz_meta = pd.read_json('meta_South_Dakota.json.gz', lines=True, compression='gzip')
biz_meta.columns = biz_meta.columns.str.lower().str.strip()
reviews_data.columns = reviews_data.columns.lower().str.strip()

print("\n" + tabulate(reviews_data.head(10), headers="keys", tablefmt="psql"))
print("\n" + tabulate(biz_meta.head(10), headers="keys", tablefmt="psql"))

# merge relevant cols from meta to reviews
keep_cols = [
    "gmap_id",        # join key
    "name",           # business name
    "category",       # type of business
    "avg_rating",     # business-level avg
    "num_of_reviews", # business popularity
    "latitude", "longitude", # optional
    "state"           # active/closed
]



# %%
