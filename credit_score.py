"""Compatibility shim: the credit score now lives in the config-driven engine (cycles.py + engine.py)."""
from engine import compute_cycle


def compute(refresh: bool = False):
    return compute_cycle("credit", refresh)
