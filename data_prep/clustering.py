import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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
