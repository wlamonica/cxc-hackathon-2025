import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def flatten(xss):
    """Flattens a list of lists"""
    return [x for xs in xss for x in xs]

def create_pair_df(data: pd.DataFrame, indicator_column = 'Indicator Code',
                   country_column = 'Country Code', year_start = 2000, year_end = 2023) -> pd.DataFrame:
    """Restructures the dataset so that columns are indicators, with (country, year) as the primary key"""
    features = data[indicator_column].unique()
    country_year_pairs = [[(c, y) for y in range(year_start, year_end + 1)] for c in data[country_column].unique()]
    country_year_pairs = flatten(country_year_pairs)
    ret_val = pd.DataFrame(country_year_pairs).rename(columns={0: 'Country', 1: 'Year'})
    
    for f in features:
        filtered = data[data[indicator_column] == f].set_index([country_column])
        vals = []
        for c, y in country_year_pairs:
            try:
                value = filtered.at[c, str(y)]
                vals.append(value if not np.isnan(value) else None)
            except KeyError:
                vals.append(None)
            
        ret_val[f] = vals
    
    return ret_val.set_index(['Country', 'Year'])

def create_countries_list(data: pd.DataFrame):
    """Generates a file containing a list of countries and their country codes"""
    cols = ['Country Name', 'Country Code']

    indicators = data[cols]
    indicators.drop_duplicates(cols, inplace=True)
    indicators.set_index('Country Code', inplace=True)

    indicators.to_excel('../created_datasets/countries.xlsx')

def create_indicators_list():
    """Generates a file containing a list of indicators with information about each one"""
    cols = ['Indicator Name', 'Indicator Code', 'short description', 'Unit of measure']

    df = pd.read_excel('../dataset/SAP_Datasets.xlsx')
    indicators = df[cols]
    indicators.drop_duplicates(subset=['Indicator Code'], inplace=True)
    indicators.set_index('Indicator Code', inplace=True)

    indicators.to_excel('../created_datasets/indicators.xlsx')


def remove_aggregate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Removes aggregate indicators from the dataset"""
    aggregate_columns = [
        "CC.EST", "PV.EST", "DT.TDS.MLAT.PG.ZS", "EG.ELC.ACCS.ZS", "EG.CFT.ACCS.ZS",
        "NY.ADJ.NNTY.PC.CD", "per_allsp.adq_pop_tot", "per_sa_allsa.adq_pop_tot",
        "SE.ADT.LITR.ZS", "SE.COM.DURS", "SE.ENR.TERT.FM.ZS", "SE.PRM.UNER.ZS",
        "SE.XPD.PRIM.ZS", "SH.ALC.PCAP.LI", "SH.DTH.COMM.ZS", "SH.H2O.BASW.ZS",
        "SH.XPD.CHEX.GD.ZS", "SI.POV.MPWB", "SL.EMP.WORK.ZS", "SL.UEM.TOTL.NE.ZS",
        "SM.POP.TOTL", "SP.DYN.CBRT.IN", "SP.DYN.WFRT", "SP.POP.DPND", "SP.POP.TOTL",
        "SP.URB.TOTL.IN.ZS"
    ]

    filtered_data = data.drop(columns=aggregate_columns, errors='ignore')
    return filtered_data


def remove_countries_with_missing_data() -> None:
    """Removes countries containing more than THRESHOLD percentage of missing data"""
    THRESHOLD = 0.7
    table = pd.read_csv('../created_datasets/output.csv')
    
    table.set_index(['Country', 'Year'], inplace=True)

    for country in np.unique([index[0] for index in table.index]):
        total_na = table.loc[(country)].isna().sum().sum()
        total_entries = table.loc[(country)].shape[0] * table.loc[(country)].shape[1]
        missing_data_percent = total_na / total_entries
        
        if missing_data_percent > THRESHOLD:
            table.drop(country, level=0, inplace=True)

    table.reset_index(inplace=True)

def cluster_data():
    """Clusters data based on some columns TODO"""
    df = pd.read_csv("../created_datasets/output.csv")

    # Some columns that might be insightful for clustering
    selected_indicators = [
        "GC.XPN.COMP.ZS",    # Government expenditure as % of GDP
        "NY.ADJ.AEDU.CD",    # Adjusted savings: Education expenditure (USD)
        "SE.ADT.LITR.FE.ZS", # Adult literacy rate, female (%)
        "SE.ADT.LITR.MA.ZS", # Adult literacy rate, male (%)
        "SH.H2O.SMDW.ZS",    # People using safely managed drinking water (%)
        "SL.UEM.ADVN.ZS",    # Unemployment, advanced education (% of labor force)
        "SP.POP.DPND.OL",    # Old-age dependency ratio
        "SP.POP.DPND.YG",    # Youth dependency ratio
    ]

    # Keep only the relevant columns
    df = df[["Country", "Year"] + selected_indicators]

    # Handle missing values (Arvin and Jake will implement LSTM to impute more precisely)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Scale the data for fair clustering (StandardScaler -> mean=0, std=1)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[selected_indicators])

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df_scaled)

    # Visualize clusters (using two indicators for a scatter plot)
    plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue="Cluster", diag_kind="kde", palette="viridis")
    plt.show()

    # sns.scatterplot(
    #     x=df["GC.XPN.COMP.ZS"],
    #     y=df["NY.ADJ.AEDU.CD"],
    #     hue=df["Cluster"],
    #     palette="viridis"
    # )
    # plt.xlabel("Government Expenditure as % of GDP")
    # plt.ylabel("Adjusted Savings: Education Expenditure (USD)")
    # plt.title("K-Means Clustering of Countries")
    # plt.legend(title="Cluster")
    # plt.tight_layout()
    # plt.show()

    # 8. Save the results with cluster labels
    df.to_csv("../created_datasets/clustered_countries.csv", index=False)
