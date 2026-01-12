
import re
from typing import Dict, Any
from dq_event import empty_dq_event


def parse_unallocated(
    *,
    test_result: str | None,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    UNALLOCATED TEST PARSER

    PASS examples:
        None
        ""

    FAIL examples:
        unallocated_records:644256, number_of_medical_units: 9, org_code:ORG_400for table: PATIENT_ADDRESS

    Normalised FAIL:
        unallocated_records:644256
        number_of_medical_units: 9
        org_code:ORG_400
        table: PATIENT_ADDRESS
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    # Add source context early to ensure it's included in all cases
    if source_context:
        event["objects"]["source"] = source_context
    if target_context:
        event["objects"]["target"] = target_context    

    # -------------------------------------------------------
    # 1. PASS CASE — no unallocated data
    # -------------------------------------------------------
    if not test_result or not isinstance(test_result, str):
        event["status"] = "pass"
        return event

    text = test_result.strip().replace("\n", " ")

    lowered = text.lower()

    if "unallocated_records" not in lowered:
        # No structural signal — treat as pass
        event["status"] = "pass"
        return event

    # -------------------------------------------------------
    # 2. FIX GLUED STRUCTURE ISSUES
    # -------------------------------------------------------
    # Fix "org_code:ORG_400for table:" → "org_code:ORG_400 for table:"
    text = re.sub(
        r"(org_code\s*:\s*[A-Z0-9_]+)\s*for\s+table\s*:",
        r"\1 for table:",
        text,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------------
    # 3. Extract unallocated metrics
    # -------------------------------------------------------
    unalloc_match = re.search(
        r"unallocated_records\s*:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    units_match = re.search(
        r"number_of_medical_units\s*:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    # -------------------------------------------------------
    # 4. Extract org_code (must appear even if broken)
    # -------------------------------------------------------
    org_code_match = re.search(
        r"org_code\s*:\s*([A-Z0-9_]+)",
        text,
        re.IGNORECASE
    )

    # -------------------------------------------------------
    # 5. Extract target table
    # -------------------------------------------------------
    table_match = re.search(
        r"for\s+table\s*:\s*([A-Za-z0-9_]+)",
        text,
        re.IGNORECASE
    )

    # -------------------------------------------------------
    # 6. Populate event["metrics"] and event["objects"]
    # -------------------------------------------------------
    event["metrics"]["unallocated_records"] = (
        int(unalloc_match.group(1)) if unalloc_match else None
    )
    event["metrics"]["number_of_medical_units"] = (
        int(units_match.group(1)) if units_match else None
    )

    event["entities"]["org_code"] = (
        org_code_match.group(1) if org_code_match else None
    )

    event["objects"]["table"] = (
        table_match.group(1).upper() if table_match else None
    )

    # -------------------------------------------------------
    # 7. Determine PASS / FAIL
    # -------------------------------------------------------
    # If unallocated_records is > 0 => FAIL
    if unalloc_match:
        rec_count = int(unalloc_match.group(1))
        if rec_count > 0:
            event["status"] = "fail"
        else:
            event["status"] = "pass"
    else:
        # No numeric signal → treat as unknown
        event["status"] = "unknown"

      

    return event


def parse_misallocated(
    *,
    test_result: str | None,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    MISALLOCATED TEST PARSER

    PASS examples:
        "0"
        "0.0"
        "misallocated records:0"

    FAIL example:
        misallocated records:4509049, misallocated org_code:ORG_768,
        number_of_medical_units:149, owner org_code:ORG_400

    Returned DQ_EVENT structure has:
        metrics:
            - misallocated_records
            - number_of_medical_units
        entities:
            - misallocated_org_code
            - owner_org_code
        status: pass/fail/unknown
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    # Add source context early to ensure it's included in all cases
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    # -------------------------------------------------------
    # 1. PASS — missing or zero
    # -------------------------------------------------------
    if not test_result or not isinstance(test_result, str):
        event["status"] = "pass"
        return event

    text = test_result.strip().replace("\n", " ")
    lowered = text.lower()

    # pure zero pass case
    if lowered in ("0", "0.0", "none"):
        event["status"] = "pass"
        return event

    # "misallocated records:0"
    zero_match = re.search(r"misallocated\s+records\s*:\s*0\b", lowered)
    if zero_match:
        event["status"] = "pass"
        return event

    # -------------------------------------------------------
    # 2. FIX any glued tokens
    # -------------------------------------------------------
    text = re.sub(
        r"(org_code\s*:\s*[A-Z0-9_]+)(?=[A-Za-z])",
        r"\1 ",
        text,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------------
    # 3. Extract metrics
    # -------------------------------------------------------
    misalloc_match = re.search(
        r"misallocated\s+records\s*:\s*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    units_match = re.search(
        r"number_of_medical_units\s*:\s*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    event["metrics"]["misallocated_records"] = (
        int(misalloc_match.group(1)) if misalloc_match else None
    )
    event["metrics"]["number_of_medical_units"] = (
        int(units_match.group(1)) if units_match else None
    )

    # -------------------------------------------------------
    # 4. Extract org codes
    # -------------------------------------------------------
    # First: misallocated org
    misalloc_org_match = re.search(
        r"misallocated\s+org_code\s*:\s*([A-Z0-9_]+)",
        text,
        flags=re.IGNORECASE
    )

    # Second: owner org
    owner_org_match = re.search(
        r"owner\s+org_code\s*:\s*([A-Z0-9_]+)",
        text,
        flags=re.IGNORECASE
    )

    event["entities"]["misallocated_org_code"] = (
        misalloc_org_match.group(1) if misalloc_org_match else None
    )

    event["entities"]["owner_org_code"] = (
        owner_org_match.group(1) if owner_org_match else None
    )

    # -------------------------------------------------------
    # 5. Determine PASS / FAIL
    # -------------------------------------------------------
    if misalloc_match:
        count = int(misalloc_match.group(1))
        event["status"] = "fail" if count > 0 else "pass"
    else:
        event["status"] = "unknown"  

    return event


def parse_allocated(
    *,
    test_result: str | None,
    test_family: str,
    test_category: str,
    severity: str | None,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
    
) -> Dict[str, Any]:
    """
    ALLOCATED TEST PARSER

    Input examples:

        FAIL:
            org_code:ORG_400 curr_alloc_count = 3581153815,
            prev_alloc_count = 29015390961

        PASS:
            org_code:ORG_511 curr_alloc_count = 40_000_000_000,
            prev_alloc_count = 39_800_000_000

    PASS condition:
        curr_alloc_count >= prev_alloc_count
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

    text = test_result.strip().replace("\n", " ")

    # -----------------------------------------
    # 1. Fix glued tokens (org_code:ORG_400curr → org_code:ORG_400 curr)
    # -----------------------------------------
    text = re.sub(
        r"(org_code\s*:\s*[A-Z0-9_]+)(?=[A-Za-z])",
        r"\1 ",
        text,
        flags=re.IGNORECASE
    )

    # -----------------------------------------
    # 2. Extract the org_code (entity)
    # -----------------------------------------
    org_match = re.search(
        r"org_code\s*:\s*([A-Z0-9_]+)",
        text,
        flags=re.IGNORECASE
    )

    event["entities"]["org_code"] = (
        org_match.group(1) if org_match else None
    )

    # -----------------------------------------
    # 3. Extract allocation metrics
    # -----------------------------------------
    curr_match = re.search(
        r"curr_alloc_count\s*=\s*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    prev_match = re.search(
        r"prev_alloc_count\s*=\s*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    curr = int(curr_match.group(1)) if curr_match else None
    prev = int(prev_match.group(1)) if prev_match else None

    event["metrics"]["curr_alloc_count"] = curr
    event["metrics"]["prev_alloc_count"] = prev

    # -----------------------------------------
    # 4. Compute ML-friendly values
    # -----------------------------------------
    if curr is not None and prev is not None:
        event["metrics"]["alloc_diff"] = curr - prev
    else:
        event["metrics"]["alloc_diff"] = None

    # -----------------------------------------
    # 5. Determine PASS / FAIL
    # -----------------------------------------
    if curr is None or prev is None:
        event["status"] = "unknown"

    elif curr >= prev:
        event["status"] = "pass"

    else:
        event["status"] = "fail"

    return event

