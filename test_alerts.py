"""Tests for alert_system module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alert_system.alerts import AlertSystem, AlertLevel, ALERT_CONFIG


@pytest.fixture
def alert_system():
    return AlertSystem()


class TestAlertSystem:
    def test_trigger_ambulance(self, alert_system, capsys):
        alert_system.trigger("ambulance_siren", "FRONT", 0.95)
        captured = capsys.readouterr()
        assert "AMBULANCE" in captured.out
        assert "CRITICAL" in captured.out

    def test_trigger_car_horn(self, alert_system, capsys):
        alert_system.trigger("car_horn", "LEFT", 0.8)
        captured = capsys.readouterr()
        assert "CAR HORN" in captured.out

    def test_priority_suppression(self, alert_system, capsys):
        # Trigger high-priority alert first
        alert_system.trigger("ambulance_siren", "FRONT", 0.95)
        # Lower-priority alert should be suppressed
        alert_system.trigger("car_horn", "LEFT", 0.8)
        captured = capsys.readouterr()
        assert "SUPPRESSED" in captured.out

    def test_clear_resets_priority(self, alert_system, capsys):
        alert_system.trigger("ambulance_siren", "FRONT", 0.95)
        alert_system.clear()
        # After clear, car horn should NOT be suppressed
        alert_system.trigger("car_horn", "LEFT", 0.8)
        captured = capsys.readouterr()
        assert "CAR HORN" in captured.out
        assert "SUPPRESSED" not in captured.out.split("CAR HORN")[-1]

    def test_history_tracking(self, alert_system):
        alert_system.trigger("police_siren", "RIGHT", 0.9)
        history = alert_system.get_history()
        assert len(history) >= 1
        assert history[0]["label"] == "police_siren"
        assert history[0]["direction"] == "RIGHT"

    def test_unknown_label_defaults(self, alert_system, capsys):
        alert_system.trigger("unknown_sound", "FRONT", 0.5)
        captured = capsys.readouterr()
        assert "AMBIENT" in captured.out

    def test_direction_display(self, alert_system, capsys):
        alert_system.trigger("truck_horn", "LEFT", 0.85)
        captured = capsys.readouterr()
        assert "LEFT" in captured.out

    def test_all_categories_have_config(self):
        expected = ["ambulance_siren", "police_siren", "truck_horn", "car_horn", "background_noise"]
        for cat in expected:
            assert cat in ALERT_CONFIG
