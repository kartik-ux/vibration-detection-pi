import numpy as np
from vibration_pi.features import extract_features, FEATURE_NAMES

def test_feature_vector_shape_and_finite():
    x=np.random.default_rng(1).normal(size=(800,3))
    features=extract_features(x,800)
    assert list(features)==FEATURE_NAMES
    assert len(features)==48
    assert np.isfinite(list(features.values())).all()
