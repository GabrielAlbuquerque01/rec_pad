SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_CLUSTERS = 3
MODEL_IDS = [f"M{i}" for i in range(1, 8)]
METRICS = ["accuracy", "precision", "sensitivity", "specificity", "roc_auc", "tn", "fp", "fn", "tp"]
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
