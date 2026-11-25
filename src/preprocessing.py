from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("constant", VarianceThreshold()),
        ("scale", StandardScaler()),
    ])
