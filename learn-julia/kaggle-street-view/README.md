Address for tutorial:

https://www.kaggle.com/competitions/street-view-getting-started-with-julia

Requires the dataset:

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

Download and untar to the `./data` directory

```
tar xafv ...
```

Split labels script into csvs

```
# Cleanup raw script
sed 's/;\(\w\)/\n\1/' list_English_Img.m > foo.m
mkdir metadata

# Split into seperate files by ']' char
csplit -q foo.m '/]/' '{*}'
for i in xx*; do mv $i metadata/$(grep -v ']' $i | head -n 1 | sed 's/list\.\(\w*\)\s=.*/\1/').txt; done

# Convert each script into a csv
for i in metadata/*; do grep -v ']' $i | sed 's/.*\[//' | sed 's/;//' | tr ' ' , > $i.csv; done

# Validate each one works
for i in metadata/*.csv; do python -c "import pandas as pd; print('$i'); pd.read_csv('$i')"; done
```
