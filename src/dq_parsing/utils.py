from dq_event import (
    build_source_context,
    build_target_context,
    empty_dq_event
)
    
from schema import (
    parse_schema_table_count,
    parse_schema_column_count, 
    parse_schema_column_names,
    parse_schema_column_position,
    parse_schema_datatypes
)

from completeness import (
    parse_completeness
)

from uniqueness import (
    parse_duplicates
)

from person import (
    parse_person_table
)

from privacy import (
    parse_sensitive_condition
)

from allocation import (
    parse_unallocated,
    parse_misallocated,
    parse_allocated
)

from referential_integrity import (
    parse_referential_integrity
)

def derive_category(name_of_test: str) -> str:
    name = name_of_test.lower()

    if "unallocated" in name:
        return "allocation_unallocated".upper()

    if "misallocated" in name:
        return "allocation_misallocated".upper()

    if "allocated" in name:
        return "allocation_allocated".upper()  

    if "schema drift" in name:
        return "schema_drift".upper()

    if "completeness" in name:
        return "completeness".upper()

    if "duplicates" in name:
        return "uniqueness".upper()

    if "referential" in name or "person table test" in name:
        return "structural_integrity".upper()

    if "sensitive" in name:
        return "sensitive_data".upper()
        
    return "other".upper()


def derive_test_family(name: str) -> str:
    n = name.lower()

    if "unallocated" in n or "misallocated" in n or "allocated" in n:
        return "allocation".upper()

    if "completeness" in n:
        return "completeness".upper()

    if "duplicate" in n:
        return "uniqueness".upper()

    if "referential" in n:
        return "referential_integrity".upper()

    if "person" in n:
        return "person".upper()    

    if "schema" in n:
        return "schema".upper()

    if "sensitive" in n: 
        return "privacy".upper()

    return "other".upper()

def parse_test_event(row):
    name = row["NAME_OF_TEST"].upper().strip()
    family = row["TEST_FAMILY"]
    category = row["TEST_CATEGORY"]
    result = str(row["TEST_RESULT"]) if row["TEST_RESULT"] is not None else ""
    severity = row["SEVERITY_LEVEL"]

    # --------------------------------------
    # SCHEMA TESTS
    # --------------------------------------
    source_context = build_source_context(
        source_database=row.get("SOURCE_DATABASE"),
        source_schema=row.get("SOURCE_SCHEMA"),
        source_table=row.get("SOURCE_TABLE"),
        source_column=row.get("SOURCE_COLUMN"),
    )

    target_context = build_target_context(
        target_database=row.get("TARGET_DATABASE"),
        target_schema=row.get("TARGET_SCHEMA"),
        target_table=row.get("TARGET_TABLE"),
        target_column=row.get("TARGET_COLUMN"),
    )
    
    if family.upper() == "SCHEMA":

        if "SCHEMA DRIFT TABLES" in name:
            return parse_schema_table_count(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        elif "SCHEMA DRIFT COLUMN COUNT" in name:
            return parse_schema_column_count(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        elif "SCHEMA DRIFT COLUMN NAME" in name:
            return parse_schema_column_names(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        elif "SCHEMA DRIFT COLUMN POSITION" in name:
            return parse_schema_column_position(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        elif "SCHEMA DRIFT DATATYPE" in name:
            return parse_schema_datatypes(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        else:
            return empty_dq_event(family)    

    if family.upper()  == "COMPLETENESS":
        return parse_completeness(
            test_result=result,
            test_family=family,
            test_category=category,
            severity=severity,
            source_context=source_context,
            target_context=target_context
        ) 

    if family.upper() == "UNIQUENESS":
        return parse_duplicates(
            test_result=result,
            test_family=family,
            test_category=category,
            severity=severity,
            source_context=source_context,
            target_context=target_context
        )

    if family.upper() == "PRIVACY":
        return parse_sensitive_condition(
            test_result=result,
            test_family=family,
            test_category=category,
            severity=severity,
            source_context=source_context,
            target_context=target_context
        )
        
    if family.upper() == "PERSON":
        return parse_person_table(
            test_result=result,
            test_family=family,
            test_category=category,
            severity=severity,
            source_context=source_context,
            target_context=target_context
        )

    if family.upper() == "ALLOCATION":

        if "UNALLOCATED" in name:
            return parse_unallocated(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

        if "MISALLOCATED" in name:
            return parse_misallocated(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )
        
        if "ALLOCATED" in name:
            return parse_allocated(
                test_result=result,
                test_family=family,
                test_category=category,
                severity=severity,
                source_context=source_context,
                target_context=target_context
            )

    if family.upper() == "REFERENTIAL_INTEGRITY":
        return parse_referential_integrity(
            test_result=result,
            test_family=family,
            test_category=category,
            severity=severity,
            source_context=source_context,
            target_context=target_context
        )        

            
    else:
        return empty_dq_event(family)

