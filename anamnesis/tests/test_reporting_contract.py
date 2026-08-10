from anamnesis.cli import _markdown_table
from anamnesis.scoring import SuccessGateResult


def _gate(*, supported: bool) -> SuccessGateResult:
    return SuccessGateResult(
        repetition=1,
        model="provider/frozen-snapshot",
        comparator="full_context",
        f1_gain=0.05 if supported else 0.04,
        input_token_reduction=0.30,
        anamnesis_false_alarm_checkpoints=0,
        comparator_false_alarm_checkpoints=0,
        f1_pass=supported,
        token_pass=True,
        false_alarm_pass=True,
        supported=supported,
    )


def test_final_markdown_states_one_overall_preregistered_conclusion() -> None:
    supported = _markdown_table([], gates=[_gate(supported=True)])
    unsupported = _markdown_table([], gates=[_gate(supported=False)])

    assert "hypothesis supported in all repetitions" in supported
    assert "hypothesis not supported by the frozen criteria" in unsupported
