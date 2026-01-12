from typing import Dict, Any

def empty_dq_event(test_family: str) -> Dict[str, Any]:
    """
    Container for all DQ test results.
    Every parser will return this structure.
    """

    return {
       "test_family": test_family,
       "test_category": None,
       "status": None,
       "metrics": {},
       "structure": {},
       "entities": {},
       "objects": {},
       "raw": {
            "original_test": None
       }
    }

def build_source_context(
    *,
    source_database: str | None,
    source_schema: str | None,
    source_table: str | None,
    source_column: str | None
) -> dict:
    # Filter out None and NaN values, convert to strings
    parts = []
    for p in [source_database, source_schema, source_table]:
        if p is not None and str(p).strip() and str(p) != 'nan':
            parts.append(str(p))

    return {
        "database": source_database,
        "schema": source_schema,
        "table": source_table,
        "column": source_column,
        "object_level": (
            "column" if source_column else 
            "table" if source_table else 
            "schema" if source_schema else 
            "database"
        ),
        "fully_qualified_name": ".".join(parts) if parts else None
    }

def build_target_context(
    *,
    target_database: str | None,
    target_schema: str | None,
    target_table: str | None,
    target_column: str | None
) -> dict:
    # Filter out None and NaN values, convert to strings
    parts = []
    for p in [target_database, target_schema, target_table]:
        if p is not None and str(p).strip() and str(p) != 'nan':
            parts.append(str(p))

    return {
        "database": target_database,
        "schema": target_schema,
        "table": target_table,
        "column": target_column,
        "object_level": (
            "column" if target_column else 
            "table" if target_table else 
            "schema" if target_schema else 
            "database"
        ),
        "fully_qualified_name": ".".join(parts) if parts else None
    }