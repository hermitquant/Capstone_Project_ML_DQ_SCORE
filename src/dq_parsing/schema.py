import re
from typing import Dict, Any
#from dq_event import empty_dq_event
from dq_event import empty_dq_event

def parse_datatype_list(text: str) -> dict:
    result = {}
    if not text:
        return result

    parts = text.split(",")
    for part in parts:
        if ":" in part:
            col, dtype = part.split(":", 1)
            result[col.strip().upper()] = dtype.strip().upper()
    return result

def parse_schema_column_count(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses schema column-count/hash-comparison test results.

    Example FAIL:
        curr_col_count=22 hash=fb3f229b...; prev_col_count=23 hash=0b7b8b...

    Example PASS:
        curr_col_count=24 hash=af67e55e...; prev_col_count=24 hash=af67e55e...
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

     # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        return event

    # ----------------------------------------
    # Extract values
    # ----------------------------------------

    # curr_col_count=N
    curr_count_match = re.search(r"curr_col_count\s*=\s*(\d+)", test_result)
    curr_count = int(curr_count_match.group(1)) if curr_count_match else None
    event["metrics"]["curr_col_count"] = curr_count

    # curr hash=<hash>
    curr_hash_match = re.search(r"curr_col_count\s*=\s*\d+\s+hash=([A-Za-z0-9]+)", test_result)
    curr_hash = curr_hash_match.group(1) if curr_hash_match else None
    event["metrics"]["curr_hash"] = curr_hash

    # prev_col_count=N
    prev_count_match = re.search(r"prev_col_count\s*=\s*(\d+)", test_result)
    prev_count = int(prev_count_match.group(1)) if prev_count_match else None
    event["metrics"]["prev_col_count"] = prev_count

    # prev hash=<hash>
    prev_hash_match = re.search(r"prev_col_count\s*=\s*\d+\s+hash=([A-Za-z0-9]+)", test_result)
    prev_hash = prev_hash_match.group(1) if prev_hash_match else None
    event["metrics"]["prev_hash"] = prev_hash

    # ----------------------------------------
    # ML-friendly metrics
    # ----------------------------------------
    if curr_count is not None and prev_count is not None:
        diff = curr_count - prev_count
        event["metrics"]["column_diff"] = diff
        event["metrics"]["columns_added"] = max(diff, 0)
        event["metrics"]["columns_removed"] = abs(diff) if diff < 0 else 0
    else:
        event["metrics"]["column_diff"] = None
        event["metrics"]["columns_added"] = None
        event["metrics"]["columns_removed"] = None

    # Hash mismatch indicator
    event["metrics"]["hash_mismatch"] = (
        1 if curr_hash and prev_hash and curr_hash != prev_hash else 0
    )

    # ----------------------------------------
    # Determine PASS/FAIL
    # ----------------------------------------
    if curr_count == prev_count and curr_hash == prev_hash:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    return event


def parse_schema_column_names(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses schema column-name comparison results.

    Handles PASS and FAIL cases.

    PASS example:
        curr_col_names=NAME:1, AGE:2 hash=aaa; prev_col_names=name:1, age:2 hash=aaa

    FAIL example:
        curr_col_names=A:1, B:2, C:3 hash=xxx; prev_col_names=A:1, B:2 hash=yyy
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        return event

    # ----------------------------------------------------------
    # Step 1 — Extract raw column name lists
    # ----------------------------------------------------------
    curr_match = re.search(r"curr_col_names\s*=\s*(.*?)\s*hash=", test_result, re.IGNORECASE)
    prev_match = re.search(r"prev_col_names\s*=\s*(.*?)\s*hash=", test_result, re.IGNORECASE)

    curr_cols_raw = curr_match.group(1) if curr_match else ""
    prev_cols_raw = prev_match.group(1) if prev_match else ""

    # ----------------------------------------------------------
    # Step 2 — Convert "NAME:1, AGE:2 …" into dict
    # ----------------------------------------------------------
    def parse_col_list(raw: str) -> Dict[str, int]:
        cols = {}
        if not raw:
            return cols

        parts = [p.strip() for p in raw.split(",") if ":" in p]
        for p in parts:
            name, pos = p.split(":", 1)
            cols[name.strip()] = int(pos.strip())
        return cols

    curr_cols = parse_col_list(curr_cols_raw)
    prev_cols = parse_col_list(prev_cols_raw)

    event["structure"]["curr_columns"] = curr_cols
    event["structure"]["prev_columns"] = prev_cols

    # ----------------------------------------------------------
    # Step 3 — Extract hashes
    # ----------------------------------------------------------
    curr_hash_match = re.search(r"curr_col_names.*?hash\s*=\s*([A-Za-z0-9]+)", test_result, re.IGNORECASE)
    prev_hash_match = re.search(r"prev_col_names.*?hash\s*=\s*([A-Za-z0-9]+)", test_result, re.IGNORECASE)

    curr_hash = curr_hash_match.group(1) if curr_hash_match else None
    prev_hash = prev_hash_match.group(1) if prev_hash_match else None

    event["metrics"]["curr_hash"] = curr_hash
    event["metrics"]["prev_hash"] = prev_hash

    # ----------------------------------------------------------
    # Step 4 — Compute differences
    # ----------------------------------------------------------
    curr_names = set(curr_cols.keys())
    prev_names = set(prev_cols.keys())

    added = curr_names - prev_names
    removed = prev_names - curr_names

    # Positional differences
    changed_positions = {
        name: (prev_cols[name], curr_cols[name])
        for name in (curr_names & prev_names)
        if prev_cols[name] != curr_cols[name]
    }

    # Case mismatches
    case_mismatches = {
        c: p
        for c in curr_names
        for p in prev_names
        if c.lower() == p.lower() and c != p
    }

    event["metrics"]["columns_added"] = sorted(list(added))
    event["metrics"]["columns_removed"] = sorted(list(removed))
    event["metrics"]["changed_positions"] = changed_positions
    event["metrics"]["case_mismatches"] = case_mismatches

    # ----------------------------------------------------------
    # Step 5 — Determine PASS / FAIL
    # ----------------------------------------------------------
    if (
        curr_hash == prev_hash
        and not added
        and not removed
        and not changed_positions
    ):
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    return event

def parse_schema_datatypes(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    # ------------------------------------------------------
    # Extract curr_datatypes, curr_hash, prev_datatypes, prev_hash
    # ------------------------------------------------------
    curr_match = re.search(
        r"curr_col_datatypes=(.+?)hash=([0-9a-f]{32})",
        test_result,
        flags=re.IGNORECASE | re.DOTALL
    )

    prev_match = re.search(
        r"prev_col_datatypes=(.+?)hash=([0-9a-f]{32})",
        test_result,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not curr_match or not prev_match:
        event["status"] = "unknown"
        return event

    curr_block, curr_hash = curr_match.group(1), curr_match.group(2)
    prev_block, prev_hash = prev_match.group(1), prev_match.group(2)

    curr_types = parse_datatype_list(curr_block)
    prev_types = parse_datatype_list(prev_block)

    # ------------------------------------------------------
    # Compare
    # ------------------------------------------------------
    added = [c for c in curr_types if c not in prev_types]
    removed = [c for c in prev_types if c not in curr_types]
    changed = {
        col: {"prev": prev_types[col], "curr": curr_types[col]}
        for col in curr_types
        if col in prev_types and curr_types[col] != prev_types[col]
    }

    # ------------------------------------------------------
    # Status (pass/fail)
    # ------------------------------------------------------
    if not added and not removed and not changed:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    # ------------------------------------------------------
    # Fill event structure
    # ------------------------------------------------------
    event["metrics"] = {
        "columns_added": added,
        "columns_removed": removed,
        "changed_datatypes": changed,
        "curr_hash": curr_hash,
        "prev_hash": prev_hash,
    }

    event["structure"] = {
        "current": curr_types,
        "previous": prev_types,
    }

    event["entities"] = {}
    event["objects"] = {}
    
    # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    
    
    return event


def parse_schema_column_position(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses schema column-position tests.

    Example PASS:
        curr_col_pos=A:1, B:2 ... hash=abcd;
        prev_col_pos=A:1, B:2 ... hash=abcd

    Example FAIL:
        curr_col_pos=A:1, B:2 ... hash=abcd;
        prev_col_pos=A:1, X:2 ... hash=ef99
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        return event

    # ----------------------------------------------------
    # Extract hash values
    # ----------------------------------------------------
    curr_hash_match = re.search(r"curr_col_pos=.*?hash=([A-Za-z0-9]+)", test_result)
    prev_hash_match = re.search(r"prev_col_pos=.*?hash=([A-Za-z0-9]+)", test_result)

    curr_hash = curr_hash_match.group(1) if curr_hash_match else None
    prev_hash = prev_hash_match.group(1) if prev_hash_match else None

    event["metrics"]["curr_hash"] = curr_hash
    event["metrics"]["prev_hash"] = prev_hash

    # ----------------------------------------------------
    # Extract column position lists
    # ----------------------------------------------------
    def extract_positions(prefix: str) -> Dict[str, int]:
        """
        Parses things like:
        curr_col_pos=A:1, B:2, C:3
        into {"A": 1, "B": 2, "C": 3}
        """
        match = re.search(rf"{prefix}\s*=\s*(.*?)\s*hash=", test_result)
        if not match:
            return {}

        segment = match.group(1)
        pairs = segment.split(",")

        pos_dict = {}
        for p in pairs:
            p = p.strip()
            if ":" not in p:
                continue
            name, pos = p.split(":", 1)
            try:
                pos_dict[name.strip()] = int(pos.strip())
            except ValueError:
                continue
        return pos_dict

    curr_positions = extract_positions("curr_col_pos")
    prev_positions = extract_positions("prev_col_pos")

    event["structure"]["curr_positions"] = curr_positions
    event["structure"]["prev_positions"] = prev_positions

    # ----------------------------------------------------
    # Compute column movement metrics
    # ----------------------------------------------------
    # Names present in curr but not prev
    added = [c for c in curr_positions if c not in prev_positions]

    # Names present in prev but not curr
    removed = [c for c in prev_positions if c not in curr_positions]

    # Names present in both but at different positions
    moved = [
        c for c in curr_positions
        if c in prev_positions and curr_positions[c] != prev_positions[c]
    ]

    event["metrics"]["columns_added"] = added
    event["metrics"]["columns_removed"] = removed
    event["metrics"]["columns_moved"] = moved

    # Numeric ML signals
    event["metrics"]["count_added"] = len(added)
    event["metrics"]["count_removed"] = len(removed)
    event["metrics"]["count_moved"] = len(moved)

    # Hash mismatch signal
    event["metrics"]["hash_mismatch"] = (
        1 if curr_hash and prev_hash and curr_hash != prev_hash else 0
    )

    # ----------------------------------------------------
    # Determine PASS / FAIL
    # ----------------------------------------------------
    if curr_hash == prev_hash:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context    

    return event

def parse_schema_table_count(
    *,
    test_result: str,
    test_family: str,
    test_category: str,
    severity: str,
    source_context: Dict[str, Any] = None,
    target_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses schema table-count/hash-comparison test results.

    FAIL example:
        curr_tbl_count=4 hash=626a...; prev_tbl_count=5 hash=c295...

    PASS example:
        curr_tbl_count=2 hash=cc4a...; prev_tbl_count=2 hash=cc4a...
    """

    event = empty_dq_event(test_family)
    event["test_category"] = test_category
    event["raw"]["original_test"] = test_result

    if not test_result or not isinstance(test_result, str):
        event["status"] = "unknown"
        return event

    # ------------------------------------------
    # Extract table counts
    # ------------------------------------------
    curr_match = re.search(r"curr_tbl_count\s*=\s*(\d+)", test_result)
    prev_match = re.search(r"prev_tbl_count\s*=\s*(\d+)", test_result)

    curr = int(curr_match.group(1)) if curr_match else None
    prev = int(prev_match.group(1)) if prev_match else None

    event["metrics"]["curr_tbl_count"] = curr
    event["metrics"]["prev_tbl_count"] = prev

    # ------------------------------------------
    # Extract hashes
    # ------------------------------------------
    curr_hash_match = re.search(r"curr_hash\s*=\s*([A-Za-z0-9]+)", test_result)
    prev_hash_match = re.search(r"prev_hash\s*=\s*([A-Za-z0-9]+)", test_result)

    curr_hash = curr_hash_match.group(1) if curr_hash_match else None
    prev_hash = prev_hash_match.group(1) if prev_hash_match else None

    event["metrics"]["curr_hash"] = curr_hash
    event["metrics"]["prev_hash"] = prev_hash

    # ------------------------------------------
    # Derived ML features
    # ------------------------------------------
    event["metrics"]["count_diff"] = (
        abs(curr - prev) if curr is not None and prev is not None else None
    )

    event["metrics"]["hash_mismatch"] = (
        1 if curr_hash and prev_hash and curr_hash != prev_hash else 0
    )

    # ------------------------------------------
    # PASS / FAIL
    # ------------------------------------------
    if curr == prev and curr_hash == prev_hash:
        event["status"] = "pass"
    else:
        event["status"] = "fail"

    # Add source context if provided
    if source_context:
        event["objects"]["source"] = source_context

    if target_context:
        event["objects"]["target"] = target_context     

    return event




