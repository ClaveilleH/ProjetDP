import pandas as pd

# Charger le CSV (adapter le nom du fichier)
df = pd.read_csv("results/CPU/result_all.csv", sep=";")

# Colonnes de temps à moyenner
time_columns = [
    "total_loading_time",
    "total_computing_time",
    "total_time"
]

# Moyenne par (size, batch_size)
mean_df = (
    df
    .groupby(["size", "batch_size"], as_index=False)[time_columns]
    .mean()
)

# Affichage
print(mean_df)

# Optionnel : sauvegarder dans un nouveau CSV
mean_df.to_csv("results/CPU/mean_times_by_size_batch.csv", sep=";", index=False)
