import gzip
import polars as pl

df = pl.DataFrame({"foo": [1, 2, 3]})
with gzip.open("bar.csv.gz", "wb") as f:
    df.write_csv(f)
