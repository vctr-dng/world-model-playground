"""Tests for the argument types shared by the command-line scripts."""

import argparse

import pytest

from world_model_playground import cli


def test_parse_resolution_accepts_square_and_rectangular_sizes():
    assert cli.parse_resolution("64") == (64, 64)
    assert cli.parse_resolution("240x320") == (240, 320)
    for invalid in ("0x64", "64x", "a", "1x2x3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_resolution(invalid)


@pytest.mark.parametrize(
    ("parse", "valid", "invalid"),
    [
        (cli.positive_int, ("1", "7"), ("0", "-1")),
        (cli.non_negative_int, ("0", "3"), ("-1",)),
        (cli.positive_float, ("0.5", "2"), ("0", "-0.1")),
        (cli.non_negative_float, ("0", "0.5"), ("-0.1",)),
        (cli.fraction, ("0", "0.3", "1"), ("-0.1", "1.1")),
    ],
)
def test_bounded_numbers_reject_values_outside_their_range(parse, valid, invalid):
    for value in valid:
        assert parse(value) == float(value)
    for value in invalid:
        with pytest.raises(argparse.ArgumentTypeError):
            parse(value)
