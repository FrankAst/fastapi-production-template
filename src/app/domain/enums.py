"""Categorical feature enumerations for the prediction inputs."""

from enum import IntEnum


class EducationLevel(IntEnum):
    REFUSED_OR_UNKNOWN = 0
    LESS_THAN_9TH_GRADE = 1
    SOME_HIGH_SCHOOL = 2
    HIGH_SCHOOL_OR_GED = 3
    SOME_COLLEGE_OR_AA = 4
    COLLEGE_GRADUATE_OR_ABOVE = 5


class DrinkingFrequency(IntEnum):
    NEVER = 0
    OCCASIONAL = 1
    MONTHLY = 2
    WEEKLY = 3
    DAILY = 4
