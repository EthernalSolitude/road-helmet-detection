"""Тесты чистой бизнес-логики определения нарушителя."""

import pytest

from violations import classify_label, no_helmet_ratio, should_flag_violator


class TestClassifyLabel:
    @pytest.mark.parametrize("name", ["helmet", "Helmet", "HELMET", "DHelmet"])
    def test_plain_helmet(self, name):
        assert classify_label(name) == "helmet"

    @pytest.mark.parametrize(
        "name",
        ["no_helmet", "NoHelmet", "no-helmet", "without_helmet", "PNoHelmet"],
    )
    def test_no_helmet_variants(self, name):
        assert classify_label(name) == "no_helmet"

    @pytest.mark.parametrize("name", ["person", "vehicle", "motorcycle", ""])
    def test_unknown_classes_are_not_counted(self, name):
        assert classify_label(name) == "unknown"


class TestNoHelmetRatio:
    def test_zero_total_returns_zero(self):
        assert no_helmet_ratio(0, 0) == 0.0

    def test_half(self):
        assert no_helmet_ratio(5, 5) == 0.5

    def test_all_no_helmet(self):
        assert no_helmet_ratio(0, 10) == 1.0

    def test_all_helmet(self):
        assert no_helmet_ratio(10, 0) == 0.0


class TestShouldFlagViolator:
    def test_below_min_observations_never_flags(self):
        # Даже 100% без шлема не считается нарушением на коротком треке
        assert should_flag_violator(0, 5, min_observations=15, threshold=0.8) is False

    def test_enough_obs_but_ratio_too_low(self):
        assert should_flag_violator(10, 5, min_observations=15, threshold=0.8) is False

    def test_ratio_exactly_at_threshold_not_flagged(self):
        # Порог строгий: ratio > threshold, равенство не считается
        assert should_flag_violator(4, 16, min_observations=15, threshold=0.8) is False

    def test_ratio_above_threshold_is_flagged(self):
        assert should_flag_violator(3, 17, min_observations=15, threshold=0.8) is True

    def test_boundary_min_observations_inclusive(self):
        # Ровно min_observations кадров — учитывается
        assert should_flag_violator(1, 14, min_observations=15, threshold=0.8) is True

    def test_zero_observations_edge(self):
        assert should_flag_violator(0, 0, min_observations=1, threshold=0.5) is False
