from typing import Any
from importlib import import_module
import logging

from time import perf_counter_ns
import json
import os


class performance_measure:
    """
    A class that measures the execution time of a code block.
    Usage:
    with performance_measure("name of code block"):
        # code block
    """

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, *args):
        self.end_time = perf_counter_ns()
        self.duration = self.end_time - self.start_time

        print(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")
        logging.info(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")


def load_module(module: str) -> Any:
    x = module.rsplit(".", 1)
    return getattr(import_module(x[0]), x[1])


def read_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def ensure_not_none(attr_name):
    """
    A decorator to ensure that the given attribute (attr_name) is not None before executing a method.
    """

    def decorator(method):
        def wrapper(instance, *args, **kwargs):
            if getattr(instance, attr_name) is None:
                raise ValueError(f"set '{attr_name}' before calling {method.__name__}")
            return method(instance, *args, **kwargs)

        return wrapper

    return decorator


def unity_coordinates_to_right_hand_coordinates(
    x: float, y: float, z: float
) -> list[float]:
    return [x, z, y]
