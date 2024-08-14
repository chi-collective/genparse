import itertools
from dataclasses import dataclass
from typing import (
    FrozenSet,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

TrainDatum = TypeVar('TrainDatum')
TestDatum = TypeVar('TestDatum')
T = TypeVar('T')


@dataclass(frozen=True, eq=True)
class Adornment:
    """Text that you put around some data.

    For few-shot modeling, a typical prefix might be 'Input: ' and suffix '\n'.
    """

    prefix: str
    suffix: str


@dataclass(frozen=True, eq=True)
class ProblemSpec:
    """Defines a set of input fields, and one output field."""

    input_fields: FrozenSet[str]
    output_field: str


@dataclass
class PromptBuilder(Generic[TrainDatum, TestDatum]):
    """Builds prompts for few-shot tasks with language models.

    Each datum has one or more input fields, and one output field. To build
    the prompt, we use the input fieldss and the output field for a set of
    training data, and the input fields for the test datum.

    For example, let's say that we're doing machine translation from French to English.
    Then we might set this up like the following:
    PromptBuilder(
        problem_spec=ProblemSpec(input_fields=frozenset(["french"]), output_field="english"),
        preamble="Let's translate from French to English!\n\n"  # A description of the task
        input_field_order=["french"],
        field_to_adornment={
            "french": Adornment("French: ", "\n"),
            "english": Adornment("English: ", "\n"),
        },
        separator="---\n",
    )

    Then when given two training datums, it would generate text like the following:
    > Let's translate from French to English!
    >
    > ---
    > French: train 1 french
    > English: train 1 english
    > ---
    > French: train 2 french
    > English: train 2 english
    > ---
    > French: test french
    > English:

    If a datum has `None` as the value for a given input field, that field is skipped.

    See tests for more examples.
    """

    problem_spec: ProblemSpec

    # Placed at the beginning of each prompt.
    preamble: Optional[str]
    input_field_order: Sequence[str]
    field_to_adornment: Mapping[str, Adornment]
    datum_adornment: Adornment = Adornment(prefix='', suffix='')
    separator: str = ''

    def __post_init__(self):
        assert self.problem_spec.input_fields == set(self.input_field_order)
        assert self.problem_spec.input_fields | {self.problem_spec.output_field} == set(
            self.field_to_adornment.keys()
        )

    def assemble(
        self, train_data: Sequence[TrainDatum], test_datum: Optional[TestDatum]
    ) -> str:
        def build_one_datum(
            datum: Union[TrainDatum, TestDatum], fields: Iterable[str]
        ) -> str:
            result_here: List[str] = []
            result_here += [self.datum_adornment.prefix]
            for field in fields:
                value = getattr(datum, field, None)
                if value is None:
                    continue
                adornment = self.field_to_adornment[field]
                result_here += [
                    adornment.prefix,
                    value,
                    adornment.suffix,
                ]
            result_here += [self.datum_adornment.suffix]
            return ''.join(result_here)

        result: List[str] = []
        if self.preamble:
            result += [self.preamble]
        for datum in train_data:
            result += [
                build_one_datum(
                    datum,
                    itertools.chain(
                        self.input_field_order, [self.problem_spec.output_field]
                    ),
                )
            ]
        if test_datum is not None:
            result += [
                build_one_datum(test_datum, self.input_field_order)
                + self.field_to_adornment[self.problem_spec.output_field].prefix
            ]
        return self.separator.join(result)

    @property
    def stop(self) -> str:
        """The string which marks the end of the output for the test datum."""
        return self.field_to_adornment[self.problem_spec.output_field].suffix

    @property
    def fixed_text_before_output(self) -> str:
        """The complete text which always appears before where the output should go."""
        return (
            self.field_to_adornment[self.input_field_order[-1]].suffix
            + self.field_to_adornment[self.problem_spec.output_field].prefix
        )

    @staticmethod
    def for_demo(do_include_context: bool, use_preamble: bool = True) -> 'PromptBuilder':
        input_field_order = (['agent_context'] if do_include_context else []) + [
            'natural'
        ]
        field_to_adornment = {
            'natural': Adornment('Human: ', '\n'),
            'canonical': Adornment('Computer: ', '\n'),
        }
        if do_include_context:
            field_to_adornment['agent_context'] = Adornment('Agent: ', '\n')
        return PromptBuilder(
            problem_spec=ProblemSpec(
                input_fields=frozenset(input_field_order), output_field='canonical'
            ),
            preamble="Let's translate what a human user says into what a computer might say.\n\n"
            if use_preamble
            else None,
            input_field_order=input_field_order,
            field_to_adornment=field_to_adornment,
            separator='\n',
        )
