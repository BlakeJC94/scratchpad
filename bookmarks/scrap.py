from pathlib import Path
from warnings import warn
import sqlite3 as sql
import shutil


# %% Connect to database

databases = [
    fp for fp in Path("~/.mozilla/firefox").expanduser().glob("**/*.sqlite") if fp.stem == "places"
]

if len(databases) == 0:
    raise FileNotFoundError("Couldn't find `places.sqlite`.")

database = databases[0]
if (n_databases := len(databases)) > 1:
    warn(f"Found {n_databases} `places.sqlite` files, using {str(database)}.")

database_cp = Path("/tmp/bkmk/foo.sqlite")  # Seems like file is locked, just copy it lol
database_cp.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(database, database_cp)

con = sql.connect(database_cp)

# %% Connect to database (simple copy)

database = Path("~/foo.sqlite").expanduser()
con = sql.connect(database)

# %% List tables in the database

query = "SELECT name FROM sqlite_master WHERE type='table';"
cursor = con.cursor()

res = cursor.execute(query).fetchall()

print(res)
# [('moz_origins',),
#  ('moz_places',),
#  ('moz_places_extra',),
#  ('moz_historyvisits',),
#  ('moz_historyvisits_extra',),
#  ('moz_inputhistory',),
#  ('moz_bookmarks',),
#  ('moz_bookmarks_deleted',),
#  ('moz_keywords',),
#  ('sqlite_sequence',),
#  ('moz_anno_attributes',),
#  ('moz_annos',),
#  ('moz_items_annos',),
#  ('moz_meta',),
#  ('moz_places_metadata',),
#  ('moz_places_metadata_search_queries',),
#  ('moz_previews_tombstones',),
#  ('sqlite_stat1',)]


# %% List columns in `moz_bookmarks`

query = "SELECT * FROM moz_bookmarks"
res = cursor.execute(query).description
print([r[0] for r in res])
# ['id',
#  'type',
#  'fk',
#  'parent',
#  'position',
#  'title',
#  'keyword_id',
#  'folder_type',
#  'dateAdded',
#  'lastModified',
#  'guid',
#  'syncStatus',
#  'syncChangeCounter']

# %% What's in here??
query = "SELECT * FROM moz_bookmarks"
res = cursor.execute(query).fetchall()
print(res)
# Seems like `dateAdded` is in epoch seconds?

# %% List columns in `moz_places`, Maybe there's a key to join on? probs `guid`?

query = "SELECT * FROM moz_places"

res = cursor.execute(query)
print("---")
print(res.fetchall())
print("---")
print([r[0] for r in res.description])
# ['id',
#  'url',
#  'title',
#  'rev_host',
#  'visit_count',
#  'hidden',
#  'typed',
#  'frecency',
#  'last_visit_date',
#  'guid',
#  'foreign_count',
#  'url_hash',
#  'description',
#  'preview_image_url',
#  'site_name',
#  'origin_id',
#  'recalc_frecency',
#  'alt_frecency',
#  'recalc_alt_frecency']

# %% What's in `moz_places?`

query = "SELECT url, title, guid FROM moz_places"

res = cursor.execute(query)
print("---")
print(res.fetchmany(5))

# %% Let's try and JOIN to test this hypothesis

query = """
SELECT *
FROM moz_places p INNER JOIN moz_bookmarks b
ON p.guid = b.guid;
"""

res = cursor.execute(query)
print("---")
print(res.fetchmany(5))
# []

# %% Well shit, that didn't work.. Verify no overlap?

q1 = "SELECT guid FROM moz_bookmarks"
res1 = cursor.execute(q1).fetchall()
res1 = {r[0] for r in res1}

q2 = "SELECT guid FROM moz_places"
res2 = cursor.execute(q2).fetchall()
res2 = {r[0] for r in res2}

print("---")
print(f"{len(res1)=}")
print(f"{len(res2)=}")
print(f"{len(res2 & res1)=}")
print("---")
# ---
# len(res1)=359
# len(res2)=8699
# len(res2 & res1)=0
# ---

# %% How about id? Pretty sure this is a local table id key though

q1 = "SELECT id FROM moz_bookmarks"
res1 = cursor.execute(q1).fetchall()
res1 = {r[0] for r in res1}

q2 = "SELECT id FROM moz_places"
res2 = cursor.execute(q2).fetchall()
res2 = {r[0] for r in res2}

print("---")
print(f"{len(res1)=}")
print(f"{len(res2)=}")
print(f"{len(res2 & res1)=}")
print("---")
# ---
# len(res1)=359
# len(res2)=8699
# len(res2 & res1)=346
# ---

# %% Bingo! Try that last idea again

query = """
SELECT p.url, b.title, b.dateAdded
FROM moz_places p INNER JOIN moz_bookmarks b
ON p.id = b.fk;
"""

res = cursor.execute(query)
print("---")
print(res.fetchmany(5))
# [('https://www.mozilla.org/privacy/firefox/', 'mobile', 1708123255251000),
# ('https://blog.inkdrop.app/how-to-set-up-neovim-0-5-modern-plugins-lsp-treesitter-etc-542c3d9c9887',
# 'How To Manage Python with Pyenv and Direnv | DigitalOcean', 1678053275812000),
# ('https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903',
# 'Programmer Interrupted: The Real Cost of Interruption and Context Switching', 1680757287482000),
# ('https://github.com/shMorganson/dot-files/blob/a858e28d1adbc0a5a7b13d8a2600c2014ec8b376/nvim/.config/nvim/lua/plugins/highlights/custom_highlights.vim',
# 'Gentle Dive into Math Behind Convolutional Neural Networks | by Piotr Skalski | Towards Data
# Science', 1634531361569000), ('https://muhammadraza.me/2022/data-oneliners/', 'CNN Explainer',
# 1634539466342000)]

# %% Nice! Now use `WHERE` and chuck it all into a DataFrame

import time
from datetime import timedelta
import pandas as pd

cur_time = int(time.time())
delta = timedelta(hours=7).total_seconds()

query = f"""
SELECT p.url, b.title, b.dateAdded
FROM moz_places p INNER JOIN moz_bookmarks b
ON p.id = b.fk
WHERE b.dateAdded > {(cur_time - delta) * 1e6};
"""

res = cursor.execute(query)
res = pd.DataFrame(res.fetchall(), columns=[r[0] for r in res.description])
print("---")
print(res)
# ---
#                                                  url                                              title         dateAdded
# 0  https://www.youtube.com/watch?v=MCs5OvhV9S4&li...  David Beazley - Python Concurrency From the Gr...  1709086750127000
# 1        https://www.youtube.com/watch?v=zduSFxRajkE            Let's build the GPT Tokenizer - YouTube  1709091384964000


# %% Alrighty take these results and sort and groupby day (just a bit of pandas munging)
#
from datetime import datetime
import pytz

res["dtAdded"] = res["dateAdded"].apply(
    lambda x: datetime.fromtimestamp(x / 1e6, tz=pytz.timezone("Australia/Melbourne"))
)
res = res.sort_values("dateAdded")

lines = []
lines.append("# Bookmarks from the last week")
lines.append("")

for k, v in res.groupby(res["dtAdded"].dt.date):
    lines.append(f"## {k}")
    lines.append("")
    for _, row in v.sort_values("dateAdded").iterrows():
        lines.append(f"[{row['title']}]({row['url']})")
        lines.append("")

print("---")
print("\n".join(lines))
print("---")
# ---
# # Bookmarks from the last week
#
# ## 2024-02-28
#
# [David Beazley - Python Concurrency From the Ground Up: LIVE! - PyCon 2015 - YouTube](https://www.youtube.com/watch?v=MCs5OvhV9S4&list=WL&index=2)
#
# [Let's build the GPT Tokenizer - YouTube](https://www.youtube.com/watch?v=zduSFxRajkE)
#
# ---

# %% Sick! That prototype works. Now let's get it working in an E2E block without requiring pandas

import argparse
import shutil
import sqlite3
import tempfile
from datetime import timedelta
from itertools import groupby
from pathlib import Path
from time import time
from typing import Dict, List, Tuple


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("firefox_dir", default="~/.mozilla/firefox")
    return parser.parse_args()


def get_database(firefox_dir: Path | str) -> Path:
    firefox_dir = Path(firefox_dir).expanduser()
    databases = [
        fp
        for fp in firefox_dir.glob("**/*.sqlite")
        if fp.stem == "places"
    ]

    if len(databases) == 0:
        raise FileNotFoundError("Couldn't find `places.sqlite`.")

    database = databases[0]
    if (n_databases := len(databases)) > 1:
        warn(f"Found {n_databases} `places.sqlite` files, using {str(database)}.")

    return database


def query_database(database: Path, delta: float | int) -> List[Tuple[str, str, int]]:
    cur_time = time()
    with sqlite3.connect(database) as con:
        cursor = con.cursor()

        query = f"""
        SELECT p.url, b.title, b.dateAdded
        FROM moz_places p INNER JOIN moz_bookmarks b
        ON p.id = b.fk
        WHERE b.dateAdded > {(cur_time - delta) * 1e6};
        """

        return cursor.execute(query).fetchall()


def format_results(results: List[Tuple[str, str, int]]) -> List[str]:
    grouped_results: Dict[str, List[Tuple[str, str, int]]] = {
        k: list(v)
        for k, v in groupby(results, key=lambda x: datetime.fromtimestamp(int(x[2] / 1e6)).date())
    }

    lines = []
    lines.append("# Bookmarks from the last week")
    lines.append("")

    for date in sorted(grouped_results.keys()):
        lines.append(f"## {date}")
        lines.append("")

        date_results = sorted(grouped_results[date], key=lambda x: x[2])
        for url, title, _ in date_results:
            lines.append(f"[{title}]({url})")
            lines.append("")

    return lines


def main():
    args = parse()
    database = get_database(args.firefox_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        database_cp = Path(tmpdirname) / "places.sqlite"
        shutil.copy(database, database_cp)
        results = query_database(database_cp, delta=timedelta(days=7).total_seconds())

    lines = format_results(results)
    print("---")
    print("\n".join(lines))
    print("---")


main()
