"""Canonical component taxonomy used by detection, datasets, and reports."""

from __future__ import annotations

CANONICAL_CLASSES = [
    "PCB",
    "cable",
    "battery",
    "ram",
    "heat_sink",
    "cpu",
    "gpu",
    "ssd",
    "hard_drive",
    "smartphone",
    "tablet",
    "display_screen",
    "keyboard",
    "mouse",
    "printer",
    "charger_power_adapter",
    "power_supply",
    "laptop_motherboard",
    "connector",
    "capacitor",
    "transformer",
    "fan",
    "electric_motor",
    "speaker",
    "optical_drive",
    "metal_component",
    "plastic_housing",
]

CLASS_ALIASES = {
    "cables": "cable",
    "circuit_boards": "PCB",
    "printed_circuit_board": "PCB",
    "pcb": "PCB",
    "phone": "smartphone",
    "mobile": "smartphone",
    "cell phone": "smartphone",
    "smartphones": "smartphone",
    "screen": "display_screen",
    "display": "display_screen",
    "lcd": "display_screen",
    "monitor": "display_screen",
    "hdd": "hard_drive",
    "hard disk": "hard_drive",
    "hard disk drive": "hard_drive",
    "heat-sink": "heat_sink",
    "heatsink": "heat_sink",
    "heat sink": "heat_sink",
    "cpu": "cpu",
    "processor": "cpu",
    "gpu": "gpu",
    "graphics_card": "gpu",
    "graphics card": "gpu",
    "ssd": "ssd",
    "solid_state_drive": "ssd",
    "solid state drive": "ssd",
    "psu": "power_supply",
    "power supply": "power_supply",
    "charger": "charger_power_adapter",
    "power adapter": "charger_power_adapter",
    "adapter": "charger_power_adapter",
    "laptop motherboard": "laptop_motherboard",
    "motherboard": "laptop_motherboard",
    "connectors": "connector",
    "ports": "connector",
    "capacitors": "capacitor",
    "transformers": "transformer",
    "cooling fan": "fan",
    "motor": "electric_motor",
    "motors": "electric_motor",
    "speakers": "speaker",
    "dvd drive": "optical_drive",
    "cd drive": "optical_drive",
    "microchip-ic": "cpu",
    "microchip_ic": "cpu",
    "passive-component": "metal_component",
    "passive_component": "metal_component",
}

FOLDER_TO_COMPONENT = {
    "Battery": "battery",
    "PCB": "PCB",
    "heat-sink": "heat_sink",
    "Microchip-IC": "cpu",
    "Passive-Component": "metal_component",
    "Resistor": "metal_component",
    "transistor": "metal_component",
    "Laptop": "laptop_motherboard",
    "Mobile": "smartphone",
    "Keyboard": "keyboard",
    "Mouse": "mouse",
    "Printer": "printer",
    "Television": "display_screen",
    "Air-Conditioner": "electric_motor",
    "Microwave": "transformer",
    "Refrigerator": "electric_motor",
    "Washing Machine": "electric_motor",
    "light bulbs": "metal_component",
}


def normalize_component(label: str) -> str:
    """Return the canonical taxonomy label for a model, folder, or user label."""
    stripped = label.strip()
    if stripped in CANONICAL_CLASSES:
        return stripped
    return CLASS_ALIASES.get(stripped.lower(), stripped)


def class_index_map() -> dict[str, int]:
    """Return YOLO class id mapping for the canonical component list."""
    return {name: index for index, name in enumerate(CANONICAL_CLASSES)}
