SQL_CORRECTION_PROMPT_TEMPLATE = """
You are a strict SQL syntax corrector. You must:
1. Analyze the BAD_SQL query.
2. Return the corrected version or the same SQL-query if it is correct and optimized in GOOD_SQL.
3. Describe what was wrong in REASON or write "nan" in REASON if query is correct.
4. Describe how you fixed it in FIX or write "The query does not need to be fixed." in FIX if query is correct.

IMPORTANT: Always return strictly the following format, without any extra text:

BAD_SQL:
{bad_sql}
GOOD_SQL:
"""


def build_sql_correction_prompt(bad_sql: str) -> str:
    """
    Builds a prompt for SQL correction using the template

    Args:
        bad_sql: The SQL query that needs to be corrected

    Returns:
        str: Formatted prompt for the model
    """
    return SQL_CORRECTION_PROMPT_TEMPLATE.format(bad_sql=bad_sql)


def build_training_prompt(bad_sql: str) -> str:
    """
    Builds a prompt for training (same as correction prompt for consistency)

    Args:
        bad_sql: The SQL query that needs to be corrected

    Returns:
        str: Formatted prompt for training
    """
    return build_sql_correction_prompt(bad_sql)


def build_training_target(good_sql: str, reason: str, fix: str) -> str:
    """
    Builds the target response for training

    Args:
        good_sql: Corrected SQL query
        reason: Reason for the error
        fix: Description of the fix

    Returns:
        str: Formatted target response
    """
    return f"GOOD_SQL:\n{good_sql}\nREASON:\n{reason}\nFIX:\n{fix}"


def parse_model_response(response: str) -> tuple:
    """
    Parses the model response to extract corrected SQL and fix description

    Args:
        response: Raw model response

    Returns:
        tuple: (corrected_sql, fix_description)
    """
    corrected_sql = ""
    fix_description = ""

    lines = response.split('\n')
    current_section = ""

    for line in lines:
        line = line.strip()
        if line.startswith("REASON:"):
            current_section = "reason"
            continue
        elif line.startswith("FIX:"):
            current_section = "fix"
            fix_description = line[4:].strip()
            continue
        elif line.startswith("GOOD_SQL:"):
            current_section = "sql"
            corrected_sql = line[9:].strip()
            continue

        # Add content to corresponding section
        if current_section == "sql" and line:
            if corrected_sql:
                corrected_sql += " " + line
            else:
                corrected_sql = line
        elif current_section == "fix" and line:
            if fix_description:
                fix_description += " " + line
            else:
                fix_description = line

    # If structured response extraction failed, try to find SQL at the beginning of response
    if not corrected_sql:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('REASON:', 'FIX:', 'BAD_SQL:', 'GOOD_SQL:')):
                corrected_sql = line
                break

        if not corrected_sql:
            corrected_sql = response.split('\n')[0].strip()

    if not fix_description:
        fix_description = "No fix description provided"

    return corrected_sql.lower(), fix_description.lower()