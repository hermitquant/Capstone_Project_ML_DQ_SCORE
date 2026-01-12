# duplicates.py
import re
from typing import Dict, Any
from dq_event import empty_dq_event

def parse_duplicates(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parser for DUPLICATES test (family = UNIQUENESS).

    Test behaviour:
        PASS → result == 0
        FAIL → result > 0

    Input:
        test_result: numeric string (e.g. "0" or "12")

    Output DQ_EVENT:
        {
           "metrics": {
               "duplicate_count": int,
               "has_duplicates": 0/1
           }
        }
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    # -----------------------------------------------
    # Normalise and parse numeric result
    # -----------------------------------------------
    try:
        duplicate_count = int(float(test_result))
    except Exception:
        event["status"] = "unknown"
        event["metrics"]["duplicate_count"] = None
        event["metrics"]["has_duplicates"] = None
        return event

    event["metrics"]["duplicate_count"] = duplicate_count
    event["metrics"]["has_duplicates"] = 1 if duplicate_count > 0 else 0

    # -----------------------------------------------
    # Determine PASS / FAIL
    # -----------------------------------------------
    if duplicate_count == 0:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    return event
