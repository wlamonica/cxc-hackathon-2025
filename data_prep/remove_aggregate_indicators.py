import pandas as pd

df = pd.read_csv('../created_datasets/output.csv')

# List of aggregate indicator columns to remove
aggregate_columns = [
    "CC.EST", "PV.EST", "DT.TDS.MLAT.PG.ZS", "EG.ELC.ACCS.ZS", "EG.CFT.ACCS.ZS",
    "NY.ADJ.NNTY.PC.CD", "per_allsp.adq_pop_tot", "per_sa_allsa.adq_pop_tot",
    "SE.ADT.LITR.ZS", "SE.COM.DURS", "SE.ENR.TERT.FM.ZS", "SE.PRM.UNER.ZS",
    "SE.XPD.PRIM.ZS", "SH.ALC.PCAP.LI", "SH.DTH.COMM.ZS", "SH.H2O.BASW.ZS",
    "SH.XPD.CHEX.GD.ZS", "SI.POV.MPWB", "SL.EMP.WORK.ZS", "SL.UEM.TOTL.NE.ZS",
    "SM.POP.TOTL", "SP.DYN.CBRT.IN", "SP.DYN.WFRT", "SP.POP.DPND", "SP.POP.TOTL",
    "SP.URB.TOTL.IN.ZS"
]

df_filtered = df.drop(columns=aggregate_columns, errors='ignore')

df_filtered.to_csv('../created_datasets/output.csv', index=False)

print("Filtered dataset saved successfully!")
