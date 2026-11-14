"""Load the shipped LLMCompass designs, single device, optionally re-split."""
import copy
import hashlib
import json

import common  # noqa: F401  (sets up LLMCompass paths)
from design_space_exploration.dse import read_architecture_template, template_to_system

DESIGNS = {
    "GA100": "configs/GA100.json",
    "generation_system": "configs/generation_system.json",
    "prefilling_system": "configs/prefilling_system.json",
}


def load_specs(design, local_KB=None, gb_MB=None):
    specs = copy.deepcopy(read_architecture_template(DESIGNS[design]))
    # Shipped configs are 4-device systems; P07 studies one device.
    specs["device_count"] = 1
    if local_KB is not None:
        specs["device"]["compute_chiplet"]["core"]["SRAM_KB"] = int(local_KB)
    if gb_MB is not None:
        # The simulator reads global_buffer_MB; the area model reads
        # physical_global_buffer_MB. Keep them equal.
        specs["device"]["io"]["global_buffer_MB"] = int(gb_MB)
        specs["device"]["io"]["physical_global_buffer_MB"] = int(gb_MB)
    return specs


def specs_key(specs):
    blob = json.dumps(specs, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def sram_of(specs):
    return (
        specs["device"]["compute_chiplet"]["core"]["SRAM_KB"],
        specs["device"]["io"]["global_buffer_MB"],
    )


def capacity_bytes(specs):
    return specs["device"]["memory"]["total_capacity_GB"] * 1e9


def make_system(specs):
    return template_to_system(specs)
