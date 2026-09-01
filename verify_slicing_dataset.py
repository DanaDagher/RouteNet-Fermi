# -*- coding: utf-8 -*-
"""
Script de vérification du dataset Zenodo 10610616 (network slicing, Farreras et al.)
avant adaptation à RouteNet-Fermi.

Chaque affirmation faite pendant l'analyse est testée ici avec un PASS/FAIL explicite.
Aucune confiance requise : lance-le et lis les résultats.

Usage :
    py -3.11 verify_slicing_dataset.py [dossier_dataset] [nb_echantillons]

Prérequis : Python >= 3.8, pip install jsonpickle numpy networkx
"""
import os
import sys
from collections import Counter, defaultdict

DATASET_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_dataset")
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 5

sys.path.insert(0, DATASET_DIR)
from datanetAPI import DatanetAPI  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def info(name, detail):
    print(f"  [INFO] {name}: {detail}")


print(f"Dataset : {DATASET_DIR}")
print(f"Échantillons analysés : {N_SAMPLES}\n")

# ---------------------------------------------------------------- structure
print("== 1. Structure des fichiers ==")
n_graphs = len(os.listdir(os.path.join(DATASET_DIR, "graphs")))
n_routings = len(os.listdir(os.path.join(DATASET_DIR, "routings")))
n_slices = len(os.listdir(os.path.join(DATASET_DIR, "slices")))
n_tars = len([f for f in os.listdir(DATASET_DIR) if f.endswith(".tar.gz")])
check("graphs/routings/slices en triplets appariés", n_graphs == n_routings == n_slices,
      f"{n_graphs}/{n_routings}/{n_slices}")
check("tarballs de résultats présents", n_tars > 0, f"{n_tars} tarballs")

# ---------------------------------------------------------------- itération API
print("\n== 2. Lecture des échantillons via datanetAPI ==")
api = DatanetAPI(DATASET_DIR, shuffle=False)
samples = []
it = iter(api)
try:
    for _ in range(N_SAMPLES):
        samples.append(next(it))
    check("l'API itère sans erreur", True, f"{len(samples)} échantillons lus")
except Exception as e:
    check("l'API itère sans erreur", False, f"{type(e).__name__}: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- attributs QoS
print("\n== 3. Configuration des files d'attente (affirmation : elle est PRÉSENTE) ==")
REQ_NODE = ["levelsQoS", "queueSizes", "schedulingPolicy", "schedulingWeights", "tosToQoSqueue"]
attrs_ok = True
policies = Counter()
policy_rule_ok = True  # WFQ sur les nœuds routeurs (inner/antenna), FIFO sur les terminaux
qsizes = Counter()
for s in samples:
    G = s.get_topology_object()
    for n in G.nodes:
        nd = G.nodes[n]
        if not all(a in nd for a in REQ_NODE):
            attrs_ok = False
        policies[nd.get("schedulingPolicy")] += 1
        expected = "WFQ" if nd.get("type") in ("inner", "antenna") else "FIFO"
        if nd.get("schedulingPolicy") != expected:
            policy_rule_ok = False
        for q in str(nd.get("queueSizes", "")).split(","):
            if q.strip():
                qsizes[q.strip()] += 1
check("tous les nœuds ont les 5 attributs QoS", attrs_ok)
check("règle : WFQ sur cœur+antennes, FIFO sur terminaux", policy_rule_ok, dict(policies))
info("tailles de files observées (paquets)", dict(qsizes.most_common(5)))

# ---------------------------------------------------------------- ToS et mapping
print("\n== 4. ToS et mapping flux -> file ==")
tos_ok, map_ok, delay_pos = True, True, True
n_flows_total = 0
bad_delay = 0
for s in samples:
    G = s.get_topology_object()
    T = s.get_traffic_matrix()
    P = s.get_performance_matrix()
    R = s.get_routing_matrix()
    N = G.number_of_nodes()
    for src in range(N):
        for dst in range(N):
            try:
                fl = T[src, dst]["Flows"]
            except Exception:
                continue
            for i, f in enumerate(fl):
                n_flows_total += 1
                if int(f["ToS"]) != src:
                    tos_ok = False
                if P[src, dst]["Flows"][i]["AvgDelay"] <= 0:
                    bad_delay += 1
                for h1 in list(R[src, dst])[:-1]:
                    qmap = [m.split(",") for m in str(G.nodes[h1]["tosToQoSqueue"]).split(";")]
                    if not any(str(int(f["ToS"])) in grp for grp in qmap):
                        map_ok = False
check("ToS == ID du nœud source (clé de mapping, pas une classe QoS)", tos_ok, f"{n_flows_total} flux")
check("chaque flux trouve sa file à chaque saut (tosToQoSqueue)", map_ok)
check("tous les delays > 0 (labels exploitables)", bad_delay == 0, f"{bad_delay}/{n_flows_total} délais <= 0")

# ---------------------------------------------------------------- percentiles
print("\n== 5. Labels de performance ==")
pcts_ok = True
pcts_present = True
for s in samples:
    T = s.get_traffic_matrix()
    P = s.get_performance_matrix()
    N = s.get_topology_object().number_of_nodes()
    for src in range(N):
        for dst in range(N):
            try:
                fl = T[src, dst]["Flows"]
            except Exception:
                continue
            for i, _ in enumerate(fl):
                pf = P[src, dst]["Flows"][i]
                if not all(k in pf for k in ("p10", "p50", "p90", "Jitter", "PktsDrop")):
                    pcts_present = False
                elif not (pf["p10"] <= pf["p50"] <= pf["p90"]):
                    pcts_ok = False
check("percentiles p10/p50/p90 + jitter + loss présents par flux", pcts_present)
check("ordre des percentiles cohérent (p10 <= p50 <= p90)", pcts_ok)
glosses = [s.get_global_losses() / max(s.get_global_packets(), 1) for s in samples]
info("taux de pertes global par échantillon", [f"{x:.1%}" for x in glosses])

# ---------------------------------------------------------------- poids WFQ
print("\n== 6. Poids WFQ (affirmation : présents, par port, somme ≈ 100) ==")
fmt = Counter()
wsum_ok = True
for s in samples:
    G = s.get_topology_object()
    for n in G.nodes:
        w = str(G.nodes[n]["schedulingWeights"])
        if w == "-":
            fmt["'-' (nœud terminal)"] += 1
        elif ";" in w:
            fmt["par port (';')"] += 1
            for port_w in w.split(";"):
                try:
                    ws = [float(x) for x in port_w.split(",")]
                    if len(ws) > 1 and max(ws) > 1 and not (99 < sum(ws) < 101):
                        wsum_ok = False
                except ValueError:
                    wsum_ok = False
        else:
            try:
                float(w.split(",")[0])
                fmt["simple"] += 1
            except ValueError:
                fmt["inconnu"] += 1
check("3 formats de poids attendus ('-', simple, par-port)", "inconnu" not in fmt, dict(fmt))
check("chaque liste de poids par port somme à ~100 (%)", wsum_ok)

# ---------------------------------------------------------------- slices
print("\n== 7. Slices ==")
join_ok = True
types_seen = Counter()
delta_bad = 0
for s in samples:
    pairs = Counter()
    for x in s.get_slices():
        types_seen[x["type"]] += 1
        if not (0 < x["delta"] <= 1):
            delta_bad += 1
        for f in x["flows"]:
            pairs[(f["origin_node"], f["destination"])] += 1
    if pairs and max(pairs.values()) > 1:
        join_ok = False
check("jointure slice<->flux par (origin,destination) unique", join_ok)
check("les 3 types de slices présents", set(types_seen) == {"eMBB", "mMTC", "URLLC"}, dict(types_seen))
check("delta dans (0,1]", delta_bad == 0, f"{delta_bad} hors bornes")

# ---------------------------------------------------------------- routage réel
print("\n== 8. Chemin réel : routing_matrix vs slice.path (test par utilisation mesurée) ==")
divergent_samples = 0
errR_total, errS_total = 0.0, 0.0
for s in samples:
    G = s.get_topology_object()
    T = s.get_traffic_matrix()
    R = s.get_routing_matrix()
    diverge = False
    loadR, loadS = defaultdict(float), defaultdict(float)
    for x in s.get_slices():
        for f in x["flows"]:
            p = eval(f["path"])  # liste texte -> liste python (données internes, pas un input utilisateur)
            src, dst = p[0], p[-1]
            rt = list(R[src, dst])
            if rt != p:
                diverge = True
            try:
                bw = T[src, dst]["Flows"][0]["AvgBw"]
            except Exception:
                continue
            for h1, h2 in zip(rt[:-1], rt[1:]):
                loadR[(h1, h2)] += bw
            for h1, h2 in zip(p[:-1], p[1:]):
                loadS[(h1, h2)] += bw
    if diverge:
        divergent_samples += 1
    for i, node in enumerate(s.port_stats):
        for j, st in node.items():
            cap = G[i][j][0]["bandwidth"] if G.has_edge(i, j) else None
            if cap is None:
                continue
            meas = st["utilization"] * cap
            errR_total += abs(meas - min(loadR[(i, j)], cap))
            errS_total += abs(meas - min(loadS[(i, j)], cap))
info("échantillons où slice.path diverge du routing", f"{divergent_samples}/{len(samples)}")
check("la charge mesurée valide ROUTING_MATRIX comme vérité terrain",
      errR_total < errS_total,
      f"erreur routing={errR_total/1e6:.0f} Mb < erreur slice.path={errS_total/1e6:.0f} Mb")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 60)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
n_fail = len(RESULTS) - n_pass
print(f"VERDICT : {n_pass} PASS / {n_fail} FAIL sur {len(RESULTS)} vérifications")
if n_fail:
    print("Échecs :")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  - {name} ({detail})")
sys.exit(1 if n_fail else 0)
