
import re
from typing import Dict, Any
from dq_event import empty_dq_event


def parse_person_table(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    PERSON TABLE TEST

    PASS examples:
        org_code:ORG_768: count_person_id: 1994142, count_distinct_person_id: 1993333
        ORG_459: count_person_id: 2233687, count_distinct_person_id: 2232599

    FAIL rule:
        count_distinct_person_id > count_person_id

    Returned structure follows the unified DQ_EVENT format.
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        return event

    text = test_result.strip()

    # ----------------------------------------
    # Extract org code (supports both formats)
    # ----------------------------------------
    # Matches:
    #   org_code:ORG_768:
    #   ORG_459:
    org_match = re.search(r"(?:org_code:)?(ORG_[0-9]+)", text, re.IGNORECASE)
    org_code = org_match.group(1).upper() if org_match else None
    event["entities"]["org_code"] = org_code

    # ----------------------------------------
    # Extract numeric metrics
    # ----------------------------------------
    count_person_match = re.search(r"count_person_id\s*:\s*(\d+)", text, re.IGNORECASE)
    count_distinct_match = re.search(r"count_distinct_person_id\s*:\s*(\d+)", text, re.IGNORECASE)

    count_person = int(count_person_match.group(1)) if count_person_match else None
    count_distinct = int(count_distinct_match.group(1)) if count_distinct_match else None

    event["metrics"]["count_person_id"] = count_person
    event["metrics"]["count_distinct_person_id"] = count_distinct

    # ----------------------------------------
    # Define PASS / FAIL
    # ----------------------------------------
    if count_person is None or count_distinct is None:
        event["status"] = "unknown"
        return event

    if count_person >= count_distinct:
        event["status"] = "pass"
    else:
        event["status"] = "fail"
        event["metrics"]["violation_amount"] = count_distinct - count_person

    return event
