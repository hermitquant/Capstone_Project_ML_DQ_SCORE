import re
from typing import Dict, Any
from dq_event import empty_dq_event


def parse_referential_integrity(
    *,
    test_result: str | None,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    REFERENTIAL INTEGRITY PARSER

    Example PASS:
        0

    Example FAIL:
        12
    → meaning 12 rows violate referential constraints.

    PASS rule:
        numeric value == 0

    FAIL rule:
        numeric value > 0
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    if test_result is None:
        event["status"] = "unknown"
        event["metrics"]["broken_fk_count"] = None
        return event

    # ------------------------------------------
    # Extract the numeric value
    # RI results are VERY simple: a number in string form
    # ------------------------------------------
    match = re.search(r"(\d+)", str(test_result))
    value = int(match.group(1)) if match else None

    event["metrics"]["broken_fk_count"] = value

    # ------------------------------------------
    # Classification
    # ------------------------------------------
    if value is None:
        event["status"] = "unknown"

    elif value == 0:
        event["status"] = "pass"

    elif value > 0:
        event["status"] = "fail"

    return event
