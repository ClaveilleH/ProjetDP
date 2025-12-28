import csv

i = "1"
input_file = f"results/GPU/{i}/results_{i}gpu.csv"
output_file = f"results/GPU/{i}/output_{i}gpu.csv"

runs = []
current_run = []

with open(input_file, newline="") as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)  # skip header

    for row in reader:
        # Détecte la séparation de run (;;;;)
        if len(row) >= 4 and all(cell == "" for cell in row[:4]):
            if current_run:
                runs.append(current_run)
                current_run = []
        else:
            current_run.append(row)

    # dernier run
    if current_run:
        runs.append(current_run)

# Agrégation des résultats
results = []

for run_id, run in enumerate(runs):
    size = run[0][1]

    total_loading = 0.0
    total_computing = 0.0
    total_time = 0.0

    for row in run:
        total_loading += float(row[2])
        total_computing += float(row[3])
        total_time += float(row[4])

    results.append([
        run_id,
        size,
        total_loading,
        total_computing,
        total_time
    ])

# Écriture du CSV résultat
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "run",
        "size",
        "total_loading_time",
        "total_computing_time",
        "total_time"
    ])
    writer.writerows(results)

print(f"Traitement terminé → {output_file}")
