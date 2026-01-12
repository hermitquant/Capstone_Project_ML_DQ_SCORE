import re
from typing import Dict, Any
from dq_event import empty_dq_event


def parse_completeness(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses COMPLETENESS test results.

    Test examples:
        "1.00" → pass
        "1"    → pass
        "0.98" → fail
        "0.47" → fail

    Completeness always returns a single numeric value
    representing ratio of non-null values.
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

        # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context  

    # ---------------------------------------------------------
    # 1. Convert the test_result into a float
    # ---------------------------------------------------------
    if test_result is None:
        event["status"] = "unknown"
        return event

    # Clean out any whitespace or stray characters
    raw_str = str(test_result).strip()

    # Extract numeric pattern (covers: 1, 1.0, 0.87, ".95")
    match = re.search(r"([0-9]*\.?[0-9]+)", raw_str)
    if not match:
        event["status"] = "unknown"
        return event

    completeness_value = float(match.group(1))
    event["metrics"]["completeness_ratio"] = completeness_value

    # ---------------------------------------------------------
    # 2. ML-friendly derived metric
    # ---------------------------------------------------------
    event["metrics"]["is_complete"] = 1 if completeness_value >= 1.0 else 0

    # ---------------------------------------------------------
    # 3. PASS / FAIL logic
    # ---------------------------------------------------------
    if completeness_value >= 1.0:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    return event
