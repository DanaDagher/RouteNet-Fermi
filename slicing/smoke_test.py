# -*- coding: utf-8 -*-
"""
Smoke test de bout en bout : dataset slicing Zenodo -> generateur adapte ->
forward pass RouteNet-Fermi adapte.

Si ce script termine avec PASS, la chaine complete est compatible :
lecture API -> hypergraphe (files WFQ par slice) -> tenseurs -> GNN -> prediction.
(Le modele n'est PAS entraine : les valeurs predites n'ont aucun sens ici,
seule la mecanique est testee.)

Usage :  ..\\venv\\Scripts\\python.exe smoke_test.py
"""
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tensorflow as tf  # noqa: E402
from data_generator import input_fn  # noqa: E402
from delay_model import RouteNet_Fermi  # noqa: E402

DATA_DIR = os.path.join(HERE, '..', 'new_dataset')

print("1/4  Construction du tf.data.Dataset...")
ds = input_fn(DATA_DIR, shuffle=False)

print("2/4  Lecture + conversion du premier echantillon (hypergraphe)...")
t0 = time.time()
features, labels = next(iter(ds))
t1 = time.time()
n_paths = int(features['traffic'].shape[0])
n_links = int(features['capacity'].shape[0])
n_queues = int(features['queue_size'].shape[0])
print(f"     OK en {t1-t0:.1f}s : {n_paths} chemins, {n_links} liens, {n_queues} files")
print(f"     slice_type presents : {sorted(set(features['slice_type'].numpy().tolist()))}")
print(f"     delta   min/max     : {float(tf.reduce_min(features['delta'])):.3f} / {float(tf.reduce_max(features['delta'])):.3f}")
print(f"     poids   min/max     : {float(tf.reduce_min(features['weight'])):.4f} / {float(tf.reduce_max(features['weight'])):.4f}")
print(f"     labels (delay) n={int(tf.size(labels))}, min={float(tf.reduce_min(labels)):.5f}, max={float(tf.reduce_max(labels)):.5f}")

print("3/4  Forward pass RouteNet-Fermi (non entraine)...")
model = RouteNet_Fermi()
t0 = time.time()
pred = model(features)
t1 = time.time()
print(f"     OK en {t1-t0:.1f}s : {int(tf.size(pred))} predictions")

print("4/4  Verifications finales...")
ok = True
if int(tf.size(pred)) != int(tf.size(labels)):
    print("     [FAIL] nombre de predictions != nombre de labels")
    ok = False
if not bool(tf.reduce_all(tf.math.is_finite(pred))):
    print("     [FAIL] predictions non finies (NaN/Inf)")
    ok = False
if ok:
    print("     [PASS] 1 prediction par flux, toutes finies")
    print(f"     exemple (5 premiers flux) pred={[round(float(x),5) for x in pred[:5]]}")
    print(f"                              vrai={[round(float(x),5) for x in labels[:5]]}")
print()
print("SMOKE TEST : " + ("PASS - chaine complete compatible" if ok else "FAIL"))
sys.exit(0 if ok else 1)
