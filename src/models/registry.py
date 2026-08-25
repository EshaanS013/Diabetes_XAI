from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


@dataclass
class ModelDefinition:
    name: str
    display_name: str
    family: str
    requires_scaling: bool
    supports_predict_proba: bool
    tuning_strategy: str  # "grid_small" | "optuna"
    explainability_strategy: str  # "linear" | "tree" | "kernel"
    estimator_factory: Callable[[int], Any]
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _logistic(seed: int) -> Any:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )


def _naive_bayes(seed: int) -> Any:
    return GaussianNB()


def _knn(seed: int) -> Any:
    return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)


def _svm(seed: int) -> Any:
    return SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )


def _random_forest(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _gradient_boosting(seed: int) -> Any:
    return GradientBoostingClassifier(random_state=seed)


def _xgboost(seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def _lightgbm(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _catboost(seed: int) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=False,
        loss_function="Logloss",
        auto_class_weights="Balanced",
    )


def _mlp(seed: int) -> Any:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=100,
        early_stopping=True,
        random_state=seed,
    )


REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        display_name="Logistic Regression",
        family="linear",
        requires_scaling=True,
        supports_predict_proba=True,
        tuning_strategy="grid_small",
        explainability_strategy="linear",
        estimator_factory=_logistic,
        search_space={"model__C": [0.01, 0.1, 1.0, 10.0]},
        notes="Interpretable linear baseline from proposal.",
    ),
    "naive_bayes": ModelDefinition(
        name="naive_bayes",
        display_name="Gaussian Naive Bayes",
        family="probabilistic",
        requires_scaling=True,
        supports_predict_proba=True,
        tuning_strategy="grid_small",
        explainability_strategy="kernel",
        estimator_factory=_naive_bayes,
        search_space={"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "knn": ModelDefinition(
        name="knn",
        display_name="K-Nearest Neighbours",
        family="instance",
        requires_scaling=True,
        supports_predict_proba=True,
        tuning_strategy="grid_small",
        explainability_strategy="kernel",
        estimator_factory=_knn,
        search_space={"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
    ),
    "svm": ModelDefinition(
        name="svm",
        display_name="Support Vector Machine",
        family="kernel",
        requires_scaling=True,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="kernel",
        estimator_factory=_svm,
        search_space={
            "model__C": ("float", 0.1, 10.0, "log"),
            "model__gamma": ("categorical", ["scale", "auto"]),
        },
        notes="Expensive on full BRFSS; subsample carefully for Phase 1 if selected.",
    ),
    "random_forest": ModelDefinition(
        name="random_forest",
        display_name="Random Forest",
        family="bagging",
        requires_scaling=False,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="tree",
        estimator_factory=_random_forest,
        search_space={
            "model__n_estimators": ("int", 100, 400),
            "model__max_depth": ("int", 6, 30),
            "model__min_samples_leaf": ("int", 1, 10),
        },
    ),
    "gradient_boosting": ModelDefinition(
        name="gradient_boosting",
        display_name="Gradient Boosting",
        family="boosting",
        requires_scaling=False,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="tree",
        estimator_factory=_gradient_boosting,
        search_space={
            "model__n_estimators": ("int", 100, 300),
            "model__learning_rate": ("float", 0.01, 0.2, "log"),
            "model__max_depth": ("int", 2, 5),
        },
    ),
    "xgboost": ModelDefinition(
        name="xgboost",
        display_name="XGBoost",
        family="boosting",
        requires_scaling=False,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="tree",
        estimator_factory=_xgboost,
        search_space={
            "model__n_estimators": ("int", 100, 500),
            "model__learning_rate": ("float", 0.01, 0.2, "log"),
            "model__max_depth": ("int", 3, 10),
            "model__subsample": ("float", 0.6, 1.0),
            "model__colsample_bytree": ("float", 0.6, 1.0),
        },
        notes="Proposal primary production candidate — must be confirmed by experiment.",
    ),
    "lightgbm": ModelDefinition(
        name="lightgbm",
        display_name="LightGBM",
        family="boosting",
        requires_scaling=False,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="tree",
        estimator_factory=_lightgbm,
        search_space={
            "model__n_estimators": ("int", 100, 500),
            "model__learning_rate": ("float", 0.01, 0.2, "log"),
            "model__num_leaves": ("int", 16, 64),
            "model__subsample": ("float", 0.6, 1.0),
        },
    ),
    "catboost": ModelDefinition(
        name="catboost",
        display_name="CatBoost",
        family="boosting",
        requires_scaling=False,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="tree",
        estimator_factory=_catboost,
        search_space={
            "model__iterations": ("int", 100, 500),
            "model__learning_rate": ("float", 0.01, 0.2, "log"),
            "model__depth": ("int", 4, 8),
        },
    ),
    "mlp": ModelDefinition(
        name="mlp",
        display_name="Multilayer Perceptron",
        family="neural",
        requires_scaling=True,
        supports_predict_proba=True,
        tuning_strategy="optuna",
        explainability_strategy="kernel",
        estimator_factory=_mlp,
        search_space={
            "model__hidden_layer_sizes": ("categorical", ["64", "64_32", "128_64"]),
            "model__alpha": ("float", 1e-5, 1e-2, "log"),
            "model__learning_rate_init": ("float", 1e-4, 1e-2, "log"),
        },
    ),
}


def get_model_definition(name: str) -> ModelDefinition:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_models() -> list[str]:
    return sorted(REGISTRY.keys())
