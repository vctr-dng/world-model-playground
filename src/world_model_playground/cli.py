"""Argument types shared by the command-line scripts."""

import argparse

__all__ = [
    "fraction",
    "non_negative_float",
    "non_negative_int",
    "parse_resolution",
    "positive_float",
    "positive_int",
]


def positive_int(value: str) -> int:
    """Parses an integer of at least 1."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Parses an integer of at least 0."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def positive_float(value: str) -> float:
    """Parses a strictly positive floating-point value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    """Parses a floating-point value of at least 0."""
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def fraction(value: str) -> float:
    """Parses a fraction in the inclusive range [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_resolution(value: str) -> tuple[int, int]:
    """Parses ``HEIGHTxWIDTH`` (or a single side for square frames) into a size."""
    try:
        sides = tuple(int(side) for side in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must look like 64x64 or 64") from error
    if len(sides) == 1:
        sides = sides * 2
    if len(sides) != 2 or min(sides) < 1:
        raise argparse.ArgumentTypeError("must look like 64x64 or 64")
    return sides
