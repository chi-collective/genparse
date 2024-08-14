from dataflow.core.lispress import try_round_trip

from bench.calflow.datum import FullDatumSub


def is_correct(pred: str, target: FullDatumSub) -> bool:
    return try_round_trip(pred) == try_round_trip(target.canonical)
