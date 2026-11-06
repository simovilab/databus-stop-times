import json

import numpy as np

from stoptimes.core.features.identity import RouteVocabulary


def test_fit_sorted_deterministic_non_numeric():
    v1 = RouteVocabulary.fit(["Red", "CR-Fitchburg", "R-12", "Red", None, float("nan")])
    v2 = RouteVocabulary.fit(["R-12", "Red", "CR-Fitchburg"])
    assert v1 == v2
    assert v1.route_ids == ("CR-Fitchburg", "R-12", "Red")
    assert len(v1) == 3


def test_encode_unseen_and_missing_are_nan():
    v = RouteVocabulary.fit(["Red", "CR-Fitchburg", "R-12"])
    out = v.encode(["Red", "Blue", None, "CR-Fitchburg", float("nan")])
    assert out.dtype == float
    assert out[0] == 2.0 and out[3] == 0.0
    assert np.isnan(out[[1, 2, 4]]).all()


def test_numeric_ids_match_their_string_form():
    v = RouteVocabulary.fit(["1", "10", "2"])
    assert list(v.encode([10, "2"])) == [1.0, 2.0]


def test_dict_round_trip_is_json_safe():
    v = RouteVocabulary.fit(["Red", "R-12"])
    again = RouteVocabulary.from_dict(json.loads(json.dumps(v.to_dict())))
    assert again == v
    assert list(again.encode(["Red"])) == list(v.encode(["Red"]))
