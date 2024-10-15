from enum import IntEnum, unique


@unique
class NodeTypesTransitionTree(IntEnum):
    INCOMING = 0
    OUTGOING = 1
    PROBABILITY = 2
    PASSING = 3
