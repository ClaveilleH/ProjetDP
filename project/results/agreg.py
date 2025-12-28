#!/usr/bin/env python3
"""
Aggrégation des résultats CPU en un CSV moyen par nb_cpu.

Le script parcourt les sous-dossiers de `results/CPU/` (ex: 1/, 4/), lit les fichiers
`results_{n}cpu.csv` (ou tout CSV présent), sépare le contenu en blocs de runs
séparés par des lignes ';;;;', calcule pour chaque run la moyenne (sur les ranks)
de `total_computing_time`, `total_loading_time`, `total_time`, puis fait la moyenne
de ces valeurs sur tous les runs trouvés pour ce `nb_cpu`.

Le fichier de sortie est `results/assets/fig1.csv` avec l'entête :
nb_cpu;computing_time;loading_time;total_time

Usage: python3 results/agreg.py
"""
import csv
import os
from statistics import mean


def read_blocks(path):
	"""Lit un fichier et retourne une liste de blocs, chaque bloc est une liste de lignes (sans header).
	Les blocs sont séparés par des lignes contenant seulement des ';' (ex: ';;;;')."""
	with open(path, 'r', encoding='utf-8') as f:
		raw = f.read().splitlines()

	blocks = []
	cur = []
	for line in raw:
		s = line.strip()
		if not s:
			continue
		# Treat lines that are only semicolons as separators
		if all(ch == ';' for ch in s):
			if cur:
				blocks.append(cur)
				cur = []
			continue
		# skip header lines
		if s.startswith('rank;') or s.startswith('rank,'):
			continue
		cur.append(s)
	if cur:
		blocks.append(cur)
	return blocks


def parse_block(block_lines):
	"""Parse un bloc et retourne (size, mean_loading, mean_computing, mean_total) ou None si vide."""
	loadings = []
	computings = []
	totals = []
	sizes = set()
	for line in block_lines:
		# accept ; or , delimiters
		parts = line.split(';') if ';' in line else line.split(',')
		if len(parts) < 5:
			continue
		try:
			rank = parts[0]
			size = int(parts[1])
			loading = float(parts[2])
			computing = float(parts[3])
			total = float(parts[4])
		except Exception:
			continue
		sizes.add(size)
		loadings.append(loading)
		computings.append(computing)
		totals.append(total)

	if not sizes or not loadings:
		return None
	# If multiple sizes in the same block, we won't mix them: take the unique size if present
	if len(sizes) > 1:
		# fallback: choose the most common size
		size = max(sizes)
	else:
		size = sizes.pop()

	return size, mean(computings), mean(loadings), mean(totals)


def aggregate_results(results_cpu_dir, out_path):
	# mapping nb_cpu -> list of runs (each run: (computing, loading, total))
	agg = {}

	for entry in os.listdir(results_cpu_dir):
		sub = os.path.join(results_cpu_dir, entry)
		if not os.path.isdir(sub):
			continue
		# look for csv files in sub
		for fname in os.listdir(sub):
			if not fname.lower().endswith('.csv'):
				continue
			fpath = os.path.join(sub, fname)
			blocks = read_blocks(fpath)
			for block in blocks:
				parsed = parse_block(block)
				if parsed is None:
					continue
				size, comp, load, tot = parsed
				agg.setdefault(size, []).append((comp, load, tot))

	# compute mean per size (average over runs)
	rows = []
	for size in sorted(agg.keys()):
		runs = agg[size]
		comps = [r[0] for r in runs]
		loads = [r[1] for r in runs]
		tots = [r[2] for r in runs]
		rows.append((size, mean(comps), mean(loads), mean(tots)))

	# ensure output dir exists
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, 'w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f, delimiter=';')
		writer.writerow(['nb_cpu', 'computing_time', 'loading_time', 'total_time'])
		for r in rows:
			writer.writerow([r[0], f"{r[1]}", f"{r[2]}", f"{r[3]}"])


if __name__ == '__main__':
	base = os.path.dirname(__file__)
	results_cpu_dir = os.path.join(base, 'GPU')
	out = os.path.join(base, 'assets', 'fig2.csv')
	aggregate_results(results_cpu_dir, out)
	print('Wrote', out)
