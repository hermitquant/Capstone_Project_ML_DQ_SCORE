# sensitive_condition.py
from typing import Dict, Any
from dq_event import empty_dq_event

def parse_sensitive_condition(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parser for SENSITIVE CONDITION test (family = PRIVACY).

    PASS condition:
        test_result == "no sensitive codes found" (case-insensitive)

    FAIL condition:
        any other string

    Returned structure matches all other DQ_EVENT parsers.
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    # Attach context
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context      

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        event["metrics"]["sensitive_flag"] = None
        return event

    cleaned = test_result.strip().lower()

    # PASS CASE
    if cleaned == "no sensitive codes found":
        event["status"] = "pass"
        event["metrics"]["sensitive_flag"] = 0
        return event

    # FAIL CASE
    event["status"] = "fail"
    event["metrics"]["sensitive_flag"] = 1
    event["structure"]["reported_sensitive_content"] = test_result

    return event
