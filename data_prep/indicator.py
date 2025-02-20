import pandas as pd

cols = ['Indicator Name', 'Indicator Code', 'short description', 'Unit of measure']

df = pd.read_excel('../dataset/SAP_Datasets.xlsx')
indicators = df[cols]
indicators.drop_duplicates(cols)
indicators.set_index('Indicator Code', inplace=True)

indicators.to_excel('../created_datasets/indicators.xlsx')
