import pandas as pd

cols = ['Country Name', 'Country Code']

df = pd.read_excel('../dataset/SAP_Datasets.xlsx')
indicators = df[cols]
indicators.drop_duplicates(cols)
indicators.set_index('Country Code', inplace=True)

indicators.to_excel('../created_datasets/countries.xlsx')