from database import insert_violation
import time


start_time = time.time()

violation_id = insert_violation(
    violation_type="LEFT BOX EMPTY",
    start_timestamp=start_time
)

print("Inserted ID:", violation_id)