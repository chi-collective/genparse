from dataclasses import dataclass
from typing import List, Optional, TextIO, TypeVar

import jsons


@dataclass(frozen=True, eq=True)
class Datum:
    dialogue_id: Optional[str]
    turn_part_index: Optional[int]
    agent_context: Optional[str]
    natural: str


@dataclass(frozen=True, eq=True)
class FullDatum(Datum):
    canonical: str


# Not contravariant since it is produced in a DataRetriever.
# Discussions at https://semanticmachines.slack.com/archives/CM88KH6EN/p1654553264411409
FullDatumSub = TypeVar('FullDatumSub', bound=FullDatum)
# Contravariant since it is ingested by either DataRetriever, DataFilter, or PromptBuilder, but never produced
DatumSub = TypeVar('DatumSub', bound=Datum, contravariant=True)


@dataclass(frozen=True, eq=True)
class BenchClampDatum:
    """
    Class to hold all possible information for each instance in BenchCLAMP. This class is used to generate, read
    and write BenchCLAMP data files. We distill it to FullDatum before using training or evaluation.
    Fields only used for CalFlow, TreeDST: last_agent_utterance, last_user_utterance, last_plan
    Fields only used for Spider and CoSQL:  schema_name, db_schema_without_val, db_schema_with_val
    """

    dialogue_id: Optional[str]
    turn_part_index: Optional[int]
    utterance: str
    plan: str
    last_agent_utterance: Optional[str] = None
    last_user_utterance: Optional[str] = None
    last_plan: Optional[str] = None
    schema_name: Optional[str] = None
    db_schema_without_val: Optional[str] = None
    db_schema_with_val: Optional[str] = None


def data_from_textio(data_file: TextIO) -> List[BenchClampDatum]:
    return [jsons.loads(line.strip(), cls=BenchClampDatum) for line in data_file]


def data_from_filename(data_filename: str):
    with open(data_filename, 'r') as f:
        return data_from_textio(f)


def transform_datum(data):
    return [
        FullDatum(
            dialogue_id=datum.dialogue_id,
            turn_part_index=datum.turn_part_index,
            natural=datum.utterance,
            canonical=datum.plan,
            agent_context='',
        )
        for datum in data
    ]
