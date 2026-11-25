from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def make_preprocessor():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("constant", VarianceThreshold()),
        ("scale", StandardScaler()),
    ])


def make_router_preprocessor():
    """Space of the revised routers: Yeo-Johnson reduces skewness before standardizing,
    so heavy tails do not become clusters of their own in K-Means or dominate distances."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("constant", VarianceThreshold()),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
