import pandas as pd

cols = ['Country Name', 'Country Code']

df = pd.read_csv('../dataset/SAP_Datasets.csv')
indicators = df[cols]
indicators.drop_duplicates(cols)
indicators.set_index('Country Code', inplace=True)

indicators.to_csv('../created_datasets/countries.csv')