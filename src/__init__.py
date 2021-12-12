from dataclasses import dataclass


@dataclass(frozen=True)
class Fact(object):
    subject: str
    predicate: str
    object: str
