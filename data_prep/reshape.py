import numpy as np
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

def createPairDF(data: pd.DataFrame, indicator_column = 'Indicator Code', country_column = 'Country Code',
                 year_start = 2000, year_end = 2023) -> pd.DataFrame:
    
    features = data[indicator_column].unique()
    country_year_pairs = [[(c, y) for y in range(year_start, year_end + 1)] for c in data[country_column].unique()]
    country_year_pairs = flatten(country_year_pairs)
    retVal = pd.DataFrame(country_year_pairs).rename(columns={0: 'Country', 1: 'Year'})
    
    for f in features:
        filtered = data[data[indicator_column] == f].set_index([country_column])
        vals = []
        for c, y in country_year_pairs:
            try:
                value = filtered.at[c, str(y)] 
                vals.append(value if not np.isnan(value) else None) 
            except KeyError as k:
                vals.append(None)
            
        retVal[f] = vals
    
    return retVal.set_index(['Country', 'Year'])