"""Unit Tests for Landmark Name Resolution and Namespace Preservation."""

import pytest
from src.metadata.lookup import LandmarkLookupTable
from src.metadata.resolver import LandmarkInfo, LandmarkResolver


def test_clean_category_url_standard():
    url = "http://commons.wikimedia.org/wiki/Category:Taj_Mahal"
    assert LandmarkResolver.clean_category_url(url) == "Taj Mahal"

    url_https = "https://commons.wikimedia.org/wiki/Category:Eiffel_Tower"
    assert LandmarkResolver.clean_category_url(url_https) == "Eiffel Tower"


def test_clean_category_url_encoded_and_special_chars():
    # Percent-encoded French accent
    url = "http://commons.wikimedia.org/wiki/Category:%C3%89cole_Militaire"
    assert LandmarkResolver.clean_category_url(url) == "École Militaire"

    # Parentheses in name
    url_parens = "http://commons.wikimedia.org/wiki/Category:Pont_Neuf_(Paris)"
    assert LandmarkResolver.clean_category_url(url_parens) == "Pont Neuf (Paris)"

    # Apostrophe encoded
    url_apostrophe = "http://commons.wikimedia.org/wiki/Category:St._Peter%27s_Basilica"
    assert LandmarkResolver.clean_category_url(url_apostrophe) == "St. Peter's Basilica"


def test_clean_category_url_invalid():
    assert LandmarkResolver.clean_category_url(None) is None
    assert LandmarkResolver.clean_category_url("") is None
    assert LandmarkResolver.clean_category_url("null") is None
    assert LandmarkResolver.clean_category_url("nan") is None


def test_resolve_landmark_with_fallback_hierarchy():
    # Missing category URL, but has hierarchical label
    info = LandmarkResolver.resolve_landmark(
        split="train",
        landmark_id="9999",
        category_url=None,
        hierarchical_label="ancient_temple_complex",
    )
    assert info.landmark_name == "ancient temple complex"
    assert info.is_resolved is True


def test_resolve_landmark_completely_unresolved():
    # Neither category nor hierarchy available: must NEVER fabricate fake names
    info = LandmarkResolver.resolve_landmark(
        split="train",
        landmark_id="8888",
        category_url=None,
        hierarchical_label=None,
    )
    assert "Unknown Landmark" in info.landmark_name
    assert "8888" in info.landmark_name
    assert info.is_resolved is False


def test_split_namespace_preservation():
    # Same numeric ID in train vs index must be treated as independent namespaces
    train_info = LandmarkResolver.resolve_landmark(
        split="train",
        landmark_id="101",
        category_url="http://commons.wikimedia.org/wiki/Category:Big_Ben",
    )
    index_info = LandmarkResolver.resolve_landmark(
        split="index",
        landmark_id="101",
        category_url="http://commons.wikimedia.org/wiki/Category:Sydney_Opera_House",
    )

    assert train_info.split == "train"
    assert index_info.split == "index"
    assert train_info.landmark_name == "Big Ben"
    assert index_info.landmark_name == "Sydney Opera House"

    # In lookup table, they must never overwrite each other
    table = LandmarkLookupTable()
    table._landmark_cache[("train", "101")] = train_info
    table._landmark_cache[("index", "101")] = index_info

    retrieved_train = table.lookup_landmark("train", "101")
    retrieved_index = table.lookup_landmark("index", "101")

    assert retrieved_train.landmark_name == "Big Ben"
    assert retrieved_index.landmark_name == "Sydney Opera House"


def test_geolocation_coordinates_separation():
    table = LandmarkLookupTable()
    info = LandmarkResolver.resolve_landmark(
        split="train",
        landmark_id="500",
        category_url="http://commons.wikimedia.org/wiki/Category:Colosseum",
    )
    table._landmark_cache[("train", "500")] = info

    # Before coordinates registered: coords must be None
    before = table.lookup_landmark("train", "500")
    assert before.latitude is None
    assert before.longitude is None

    # Register coordinates
    table.register_coordinates("train", "500", latitude=41.8902, longitude=12.4922)
    after = table.lookup_landmark("train", "500")
    assert after.latitude == pytest.approx(41.8902, abs=1e-4)
    assert after.longitude == pytest.approx(12.4922, abs=1e-4)

    # Invalid coordinates out of range should raise ValueError
    with pytest.raises(ValueError):
        table.register_coordinates("train", "500", latitude=95.0, longitude=10.0)
