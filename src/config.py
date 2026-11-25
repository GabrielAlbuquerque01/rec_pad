"""Experimental parameters, fixed before the confirmatory run (docs/PLANO.md)."""

# Each repetition reshuffles the outer and inner partitions and reseeds the stochastic models.
REPETITION_SEEDS = (11, 22, 33)
OUTER_FOLDS = 5
INNER_FOLDS = 3
KMEANS_INIT = 20
ALPHA = 0.05

# Original local hybrid (M7): fixed number of regions on standardized attributes.
N_CLUSTERS = 3
# Revised local hybrids (M9, M10): candidates chosen by cross-fitting on the inner folds.
CLUSTER_CANDIDATES = (1, 2, 3, 4, 5, 6)
NEIGHBOR_CANDIDATES = (7, 15, 31)
SHRINKAGE_CANDIDATES = (0, 10, 30)

BASE_MODELS = ["M1", "M2", "M3", "M4", "M5"]
HYBRID_MODELS = ["M6", "M7", "M8", "M9", "M10"]
MODEL_IDS = BASE_MODELS + HYBRID_MODELS
MODEL_NAMES = {
    "M1": "SVM linear", "M2": "SVM-RBF", "M3": "MLP", "M4": "GaussianNB", "M5": "KNN",
    "M6": "Global (aula)", "M7": "Local K-Means (original)",
    "M8": "Global v2", "M9": "Local K-Means v2", "M10": "Local k-NN v2",
}
# Main Friedman-Nemenyi comparison: individual models and revised hybrids.
PRIMARY_MODELS = BASE_MODELS + ["M8", "M9", "M10"]
CONTROL_MODEL = "M9"
# (proposed, reference): effect of each change, tested apart from the main comparison.
ABLATION_PAIRS = [("M8", "M6"), ("M9", "M7"), ("M10", "M8")]

METRICS = ["accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "roc_auc",
           "tn", "fp", "fn", "tp"]
# Raw labels are explicit; numerical labels are converted to strings on loading.
DATASETS = {
    "sonar": (151, 208, 60, {"R": 0, "M": 1}),
    "ionosphere": (52, 351, 34, {"b": 0, "g": 1}),
    "banknote": (267, 1372, 4, {"0": 0, "1": 1}),
    "blood": (176, 748, 4, {"0": 0, "1": 1}),
    "climate": (252, 540, 18, {"0": 0, "1": 1}),
    "qsar": (254, 1055, 41, {"NRB": 0, "RB": 1}),
    "raisin": (850, 900, 7, {"Kecimen": 0, "Besni": 1}),
    "rice": (545, 3810, 7, {"Cammeo": 0, "Osmancik": 1}),
    "spambase": (94, 4601, 57, {"0": 0, "1": 1}),
    "retinopathy": (329, 1151, 19, {"0": 0, "1": 1}),
}
