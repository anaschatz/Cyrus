from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
import uuid
from datetime import timedelta

from jarvis_orchestrator.scheduler_contract import (
    MAXIMUM_DUE_AT_UTC, PackagedTimezoneProvider, ScheduleContractError, compile_proposal,
    ensure_minimum_lead,
)
from test_scheduler_v5 import proposal_input, proposal_output, temporal_none


class SchedulerContractBoundaryTests(unittest.TestCase):
    def compile(self, expression, *, received="2030-01-10T12:00:00.000Z", recurrence="NONE"):
        request_id = str(uuid.uuid4())
        incoming = proposal_input(request_id, received)
        incoming["default_timezone"] = "UTC"
        outgoing = proposal_output(request_id, recurrence=recurrence)
        outgoing["time_expression"] = expression
        return compile_proposal(incoming, outgoing, compiler_qualification_sha256="1" * 64,
                                zones=PackagedTimezoneProvider())

    def test_yearless_impossible_dates_and_exhausted_year_range_do_not_hang(self):
        script = textwrap.dedent("""\
            from test_scheduler_contract import SchedulerContractBoundaryTests, temporal_none
            from jarvis_orchestrator.scheduler_contract import ScheduleContractError
            case = SchedulerContractBoundaryTests()
            for month, day, received in ((2, 30, '2030-01-10T12:00:00.000Z'),
                                         (4, 31, '2030-01-10T12:00:00.000Z'),
                                         (1, 1, '9999-12-31T12:00:00.000Z')):
                expression = {**temporal_none(), 'kind': 'CALENDAR_DATE', 'month': month,
                              'day': day, 'local_time': '12:00:00.000'}
                try:
                    case.compile(expression, received=received)
                except ScheduleContractError:
                    pass
                else:
                    raise AssertionError('invalid date accepted')
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_yearless_leap_date_selects_next_valid_year(self):
        compiled = self.compile({**temporal_none(), "kind": "CALENDAR_DATE", "month": 2,
                                 "day": 29, "local_time": "12:00:00.000"})
        self.assertEqual(compiled.first_due_at_utc, "2032-02-29T12:00:00.000Z")

    def test_yearless_dst_gap_requires_clarification_instead_of_changing_year(self):
        expression = {**temporal_none(), "kind": "CALENDAR_DATE", "month": 3, "day": 31,
                      "local_time": "03:30:00.000", "timezone": "Europe/Athens"}
        with self.assertRaisesRegex(ScheduleContractError, "one_shot_dst_gap"):
            self.compile(expression, received="2030-03-01T00:00:00.000Z")
        compiled = self.compile(expression, received="2030-03-01T00:00:00.000Z", recurrence="DAILY")
        self.assertEqual(compiled.first_due_at_utc, "2030-04-01T00:30:00.000Z")

    def test_out_of_range_time_arithmetic_has_a_contract_error(self):
        expressions = (
            ({**temporal_none(), "kind": "ABSOLUTE_LOCAL", "local_datetime": "9999-12-30T09:00:00.000"},
             "9999-12-29T00:00:00.000Z"),
            ({**temporal_none(), "kind": "AFTER_DURATION", "amount": 10_000_000, "unit": "WEEK"},
             "2030-01-10T12:00:00.000Z"),
            ({**temporal_none(), "kind": "RELATIVE_DAY", "relation": "TOMORROW", "local_time": "12:00:00.000"},
             "9999-12-31T12:00:00.000Z"),
        )
        for expression, received in expressions:
            with self.subTest(kind=expression["kind"]), self.assertRaises(ScheduleContractError):
                self.compile(expression, received=received)

        def stamp(value):
            return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        reference = MAXIMUM_DUE_AT_UTC - timedelta(seconds=10)
        ensure_minimum_lead(stamp(MAXIMUM_DUE_AT_UTC), reference)
        with self.assertRaisesRegex(ScheduleContractError, "invalid_time_range"):
            ensure_minimum_lead(stamp(MAXIMUM_DUE_AT_UTC + timedelta(milliseconds=1)), reference)

    def test_target_revision_cannot_use_boolean_or_noninteger_equality(self):
        request_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
        incoming = proposal_input(request_id, "2030-01-10T12:00:00.000Z")
        incoming["selected_target"].update(job_id=job_id, job_revision=1)
        outgoing = proposal_output(request_id)
        outgoing.update(operation="CANCEL_JOB", target_job_id=job_id,
                        target_job_revision=1, time_expression=temporal_none())
        outgoing["notification"].update(title=None, body=None)
        valid = compile_proposal(incoming, outgoing, compiler_qualification_sha256="1" * 64,
                                 zones=PackagedTimezoneProvider())
        self.assertEqual(valid.proposal["target_job_revision"], 1)
        for revision in (True, 1.0, "1", [], {}):
            with self.subTest(revision=revision), self.assertRaises(ScheduleContractError):
                compile_proposal(incoming, {**outgoing, "target_job_revision": revision},
                                 compiler_qualification_sha256="1" * 64, zones=PackagedTimezoneProvider())

        incoming["selected_target"]["job_revision"] = 2**63
        outgoing["target_job_revision"] = 2**63
        with self.assertRaises(ScheduleContractError):
            compile_proposal(incoming, outgoing, compiler_qualification_sha256="1" * 64,
                             zones=PackagedTimezoneProvider())

    def test_malformed_boundary_values_raise_contract_errors(self):
        request_id = str(uuid.uuid4())
        for field, invalid in (("default_locale", []), ("default_locale", {}),
                               ("request_text", "invalid-\ud800"),
                               ("request_received_at_utc", "2030-02-30T00:00:00.000Z")):
            incoming = proposal_input(request_id, "2030-01-10T12:00:00.000Z")
            incoming[field] = invalid
            with self.subTest(input=field), self.assertRaises(ScheduleContractError):
                compile_proposal(incoming, proposal_output(request_id),
                                 compiler_qualification_sha256="1" * 64, zones=PackagedTimezoneProvider())
        for container, field in (("notification", "locale"), ("notification", "visibility"),
                                 ("recurrence", "interval_unit"), (None, "clarification_reason")):
            outgoing = proposal_output(request_id)
            (outgoing if container is None else outgoing[container])[field] = []
            with self.subTest(output=field), self.assertRaises(ScheduleContractError):
                compile_proposal(proposal_input(request_id, "2030-01-10T12:00:00.000Z"), outgoing,
                                 compiler_qualification_sha256="1" * 64, zones=PackagedTimezoneProvider())

    def test_recurrence_until_cannot_precede_its_first_occurrence(self):
        request_id = str(uuid.uuid4())
        incoming = proposal_input(request_id, "2030-01-10T12:00:00.000Z")
        outgoing = proposal_output(request_id, recurrence="DAILY", minutes=20)
        outgoing["recurrence"]["until"] = {
            **temporal_none(), "kind": "AFTER_DURATION", "amount": 10, "unit": "MINUTE",
        }
        with self.assertRaisesRegex(ScheduleContractError, "conflicting_fields"):
            compile_proposal(incoming, outgoing, compiler_qualification_sha256="1" * 64,
                             zones=PackagedTimezoneProvider())
        outgoing["recurrence"]["until"]["amount"] = 20
        compiled = compile_proposal(incoming, outgoing, compiler_qualification_sha256="1" * 64,
                                    zones=PackagedTimezoneProvider())
        self.assertEqual(compiled.first_due_at_utc, compiled.proposal["schedule"]["until_at_utc"])

    def test_a_proposal_requires_an_actual_operation(self):
        request_id = str(uuid.uuid4())
        outgoing = proposal_output(request_id)
        outgoing.update(operation="NONE", time_expression=temporal_none())
        outgoing["notification"].update(title=None, body=None)
        with self.assertRaises(ScheduleContractError):
            compile_proposal(proposal_input(request_id, "2030-01-10T12:00:00.000Z"), outgoing,
                             compiler_qualification_sha256="1" * 64, zones=PackagedTimezoneProvider())


if __name__ == "__main__":
    unittest.main()
