import pickle
from evidence_forecast.pipeline_layer import PipelineFeatures
from evidence_forecast.flip_forecaster import predict_flip
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def test_pipeline_features_map_to_forecaster_input_names(tmp_path, seed_env):
    features = [
        "ci_width", "pi_width", "distance_from_null",
        "k", "tau2", "i2",
        "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
        "pipeline_trial_count", "pipeline_expected_n",
        "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
    ]
    pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("clf", DummyClassifier(strategy="prior"))])
    pipe.fit(np.zeros((10, len(features))), np.r_[[0]*5, [1]*5])
    path = tmp_path / "m.pkl"
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipe, "features": features, "schema_version": "1.0.0"}, f)

    pf = PipelineFeatures(3, 10000, 0.05, 1.2, 0.7, False)
    inputs = dict(
        ci_width=0.3, pi_width=0.8, distance_from_null=0.1,
        k=8, tau2=0.04, i2=0.5,
        fragility_index=5, egger_p=0.2, trim_fill_delta=0.02, benford_mad=0.015,
        pipeline_trial_count=pf.trial_count,
        pipeline_expected_n=pf.expected_n_sum,
        pipeline_sponsor_entropy=pf.sponsor_entropy,
        pipeline_design_het=pf.design_heterogeneity,
        pipeline_empty=pf.pipeline_empty,
    )
    # If any PipelineFeatures field was silently renamed, this call would raise
    f = predict_flip(inputs, model_path=path, bootstrap_n=10, seed=0)
    assert 0.0 <= f.probability <= 1.0
