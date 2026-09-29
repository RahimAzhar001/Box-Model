
from database import update_violation
from datetime import datetime


# =========================================================
# TEST UPDATE FUNCTION
# =========================================================

# Use an EXISTING violation ID from MySQL
VIOLATION_ID = 3


# ---------------------------------------------------------
# Test end time
# ---------------------------------------------------------

end_datetime = datetime(
    2026,
    9,
    26,
    2,
    35,
    30
)

end_timestamp = end_datetime.timestamp()


# ---------------------------------------------------------
# Test duration
# 5 minutes 15 seconds
# ---------------------------------------------------------

duration_seconds = 5 * 60 + 15


# ---------------------------------------------------------
# Call database update
# ---------------------------------------------------------

result = update_violation(
    violation_id=VIOLATION_ID,
    end_timestamp=end_timestamp,
    duration_seconds=duration_seconds
)


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

if result:
    print("\nTEST PASSED")
else:
    print("\nTEST FAILED")

