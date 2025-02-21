import pandas as pd
import numpy as np

THRESHOLD = 0.7
table = pd.read_csv('../created_datasets/output.csv')

def remove_countries_with_missing_data() -> None:
    """Removes countries containing more than THRESHOLD percentage of missing data"""
    table.set_index(['Country', 'Year'], inplace=True)

    for country in np.unique([index[0] for index in table.index]):
        total_na = table.loc[(country)].isna().sum().sum()
        total_entries = table.loc[(country)].shape[0] * table.loc[(country)].shape[1]
        missing_data_percent = total_na / total_entries
        
        if missing_data_percent > THRESHOLD:
            table.drop(country, level=0, inplace=True)

    table.reset_index(inplace=True)
remove_countries_with_missing_data()
table.to_csv('../created_datasets/removed_countries.csv')
