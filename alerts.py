"""
alert_system.alerts — Console-based visual alert system for VibeDrive.

Renders color-coded, priority-based alerts using ANSI escape codes.
Emergency vehicles always override lower-priority alerts.
"""

import os
import sys

# Try to use colorama for Windows ANSI support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


class AlertLevel:
    """Alert priority levels and their visual properties."""

    CRITICAL = 1  # Ambulance
    HIGH = 2      # Police
    MEDIUM = 3    # Truck horn
    LOW = 4       # Car horn
    INFO = 5      # Background noise


# Alert configuration per sound category
ALERT_CONFIG = {
    "ambulance_siren": {
        "level": AlertLevel.CRITICAL,
        "label": "AMBULANCE SIREN",
        "color": "RED",
        "symbol": "🚑",
        "message": "EMERGENCY — PULL OVER IMMEDIATELY",
    },
    "police_siren": {
        "level": AlertLevel.HIGH,
        "label": "POLICE SIREN",
        "color": "BLUE",
        "symbol": "🚔",
        "message": "POLICE VEHICLE NEARBY — YIELD",
    },
    "truck_horn": {
        "level": AlertLevel.MEDIUM,
        "label": "TRUCK HORN",
        "color": "ORANGE",
        "symbol": "🚛",
        "message": "HEAVY VEHICLE WARNING",
    },
    "car_horn": {
        "level": AlertLevel.LOW,
        "label": "CAR HORN",
        "color": "YELLOW",
        "symbol": "🚗",
        "message": "VEHICLE HORN DETECTED",
    },
    "background_noise": {
        "level": AlertLevel.INFO,
        "label": "AMBIENT",
        "color": "GRAY",
        "symbol": "🔇",
        "message": "Normal traffic noise",
    },
}

# ANSI color codes
ANSI_COLORS = {
    "RED":    "\033[91m",
    "BLUE":   "\033[94m",
    "ORANGE": "\033[38;5;208m",
    "YELLOW": "\033[93m",
    "GRAY":   "\033[90m",
    "WHITE":  "\033[97m",
    "RESET":  "\033[0m",
    "BOLD":   "\033[1m",
    "BG_RED":    "\033[41m",
    "BG_BLUE":   "\033[44m",
    "BG_ORANGE": "\033[48;5;208m",
    "BG_YELLOW": "\033[43m",
    "BG_GRAY":   "\033[100m",
}

# Direction arrow art
DIRECTION_ARROWS = {
    "LEFT":   "◄◄◄ LEFT",
    "RIGHT":  "RIGHT ►►►",
    "FRONT":  "▲▲▲ FRONT",
    "BEHIND": "▼▼▼ BEHIND",
}

# Priority name mapping
PRIORITY_NAMES = {
    1: "CRITICAL",
    2: "HIGH",
    3: "MEDIUM",
    4: "LOW",
    5: "INFO",
}


class AlertSystem:
    """Visual alert system with priority management."""

    def __init__(self):
        self.active_alert_level = AlertLevel.INFO
        self.alert_history = []

    def _colorize(self, text, color_name):
        """Apply ANSI color to text."""
        color = ANSI_COLORS.get(color_name, "")
        reset = ANSI_COLORS["RESET"]
        return f"{color}{text}{reset}"

    def _make_alert_box(self, config, direction, confidence):
        """Create a formatted alert box string."""
        color = config["color"]
        bg_color = f"BG_{color}"
        bold = ANSI_COLORS["BOLD"]
        reset = ANSI_COLORS["RESET"]
        fg = ANSI_COLORS.get(color, "")
        bg = ANSI_COLORS.get(bg_color, "")

        width = 56
        border = fg + "═" * width + reset
        thin_border = fg + "─" * width + reset

        direction_display = DIRECTION_ARROWS.get(direction, direction)
        priority_name = PRIORITY_NAMES.get(config["level"], "UNKNOWN")
        conf_bar_len = int(confidence * 20)
        conf_bar = "█" * conf_bar_len + "░" * (20 - conf_bar_len)

        lines = [
            "",
            fg + "╔" + "═" * width + "╗" + reset,
            fg + "║" + reset + bg + bold + f" {config['symbol']}  {config['label']:^{width - 6}} " + reset + fg + "║" + reset,
            fg + "╠" + "═" * width + "╣" + reset,
            fg + "║" + reset + f"  Priority:   {bold}{fg}{priority_name}{reset}" + " " * (width - 16 - len(priority_name)) + fg + "║" + reset,
            fg + "║" + reset + f"  Direction:  {bold}{direction_display}{reset}" + " " * (width - 14 - len(direction_display)) + fg + "║" + reset,
            fg + "║" + reset + f"  Confidence: {conf_bar} {confidence * 100:5.1f}%" + " " * (width - 37) + fg + "║" + reset,
            fg + "╠" + "═" * width + "╣" + reset,
            fg + "║" + reset + f"  {bold}{config['message']}{reset}" + " " * (width - 2 - len(config['message'])) + fg + "║" + reset,
            fg + "╚" + "═" * width + "╝" + reset,
            "",
        ]
        return "\n".join(lines)

    def trigger(self, label, direction="FRONT", confidence=0.0):
        """
        Trigger an alert for a detected sound.

        Emergency vehicles override lower-priority alerts.

        Parameters:
            label      (str):   Sound category label
            direction  (str):   Direction string (LEFT/RIGHT/FRONT/BEHIND)
            confidence (float): Classification confidence (0-1)
        """
        config = ALERT_CONFIG.get(label)
        if config is None:
            config = ALERT_CONFIG["background_noise"]

        # Emergency vehicle priority: override if new alert is higher priority
        if config["level"] <= self.active_alert_level:
            self.active_alert_level = config["level"]

            alert_box = self._make_alert_box(config, direction, confidence)
            print(alert_box)

            # Record in history
            self.alert_history.append({
                "label": label,
                "direction": direction,
                "confidence": confidence,
                "priority": config["level"],
            })
        else:
            # Lower priority than current active alert
            fg = ANSI_COLORS.get(config["color"], "")
            reset = ANSI_COLORS["RESET"]
            print(
                f"  {fg}[SUPPRESSED]{reset} {config['symbol']} {config['label']} "
                f"(Priority {config['level']} < Active {self.active_alert_level})"
            )

    def clear(self):
        """Reset active alert level."""
        self.active_alert_level = AlertLevel.INFO

    def get_history(self):
        """Get list of triggered alerts."""
        return self.alert_history
