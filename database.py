import mysql.connector
from mysql.connector import Error
from datetime import datetime, time

from config import (
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_NAME,
    DB_PORT
)


# =========================================================
# GET DATABASE CONNECTION
# =========================================================

def get_connection():

    try:

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():

            return connection

    except Error as e:

        print(
            f"[DATABASE] Connection error: {e}"
        )

    return None





# =========================================================
# INSERT VIOLATION
# =========================================================

def insert_violation(
    violation_type,
    start_timestamp
):

    connection = get_connection()

    if connection is None:

        print(
            "[DATABASE] Unable to insert violation."
        )

        return None

    cursor = None

    try:

        cursor = connection.cursor()

        # -------------------------------------------------
        # Convert Unix timestamp to datetime
        # -------------------------------------------------

        start_datetime = datetime.fromtimestamp(
            start_timestamp
        )

        violation_date = start_datetime.date()

        start_time = start_datetime.time()

        # -------------------------------------------------
        # INSERT QUERY
        # -------------------------------------------------

        query = """
            INSERT INTO tbl_material_violations
            (
                violation_date,
                violation_type,
                start_time
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
        """

        values = (
            violation_date,
            violation_type,
            start_time
        )

        cursor.execute(
            query,
            values
        )

        connection.commit()

        # -------------------------------------------------
        # GET GENERATED ID
        # -------------------------------------------------

        violation_id = cursor.lastrowid

        print(
            f"[DATABASE] Violation inserted. "
            f"ID: {violation_id}"
        )

        print(
            f"[DATABASE] Type: {violation_type}"
        )

        print(
            f"[DATABASE] Start: "
            f"{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return violation_id

    except Error as e:

        print(
            f"[DATABASE] Insert error: {e}"
        )

        connection.rollback()

        return None

    finally:

        if cursor is not None:

            cursor.close()

        if connection is not None and connection.is_connected():

            connection.close()


# # =========================================================
# # UPDATE VIOLATION
# # =========================================================

# def update_violation(
#     violation_id,
#     end_timestamp,
#     duration_seconds
# ):

#     if violation_id is None:

#         print(
#             "[DATABASE] Invalid violation ID."
#         )

#         return False

#     connection = get_connection()

#     if connection is None:

#         print(
#             "[DATABASE] Unable to update violation."
#         )

#         return False

#     cursor = None

#     try:

#         cursor = connection.cursor()

#         # -------------------------------------------------
#         # Convert Unix timestamp to datetime
#         # -------------------------------------------------

#         end_datetime = datetime.fromtimestamp(
#             end_timestamp
#         )

#         end_time = end_datetime.time()

#         # -------------------------------------------------
#         # Convert duration seconds to TIME
#         # -------------------------------------------------

#         duration_seconds = int(
#             max(
#                 0,
#                 duration_seconds
#             )
#         )

#         hours = int(
#             duration_seconds // 3600
#         )

#         minutes = int(
#             (duration_seconds % 3600) // 60
#         )

#         seconds = int(
#             duration_seconds % 60
#         )

#         # MySQL TIME supports values greater than
#         # 24 hours, but Python datetime.time does not.
#         #
#         # For the current use case this keeps the value
#         # within a normal 24-hour TIME.

#         duration = time(
#             hour=hours % 24,
#             minute=minutes,
#             second=seconds
#         )

#         # -------------------------------------------------
#         # UPDATE QUERY
#         # -------------------------------------------------

#         query = """
#             UPDATE tbl_material_violations

#             SET
#                 end_time = %s,
#                 duration = %s

#             WHERE id = %s
#         """

#         values = (
#             end_time,
#             duration,
#             violation_id
#         )

#         cursor.execute(
#             query,
#             values
#         )

#         connection.commit()
        
#     finally:

#         if cursor is not None:

#             cursor.close()

#         if connection is not None and connection.is_connected():

#             connection.close()        

# =========================================================
# UPDATE VIOLATION
# =========================================================

def update_violation(
    violation_id,
    end_timestamp,
    duration_seconds
):

    if violation_id is None:

        print(
            "[DATABASE] Invalid violation ID."
        )

        return False

    connection = get_connection()

    if connection is None:

        print(
            "[DATABASE] Unable to update violation."
        )

        return False

    cursor = None

    try:

        cursor = connection.cursor()

        # -------------------------------------------------
        # Convert Unix timestamp to datetime
        # -------------------------------------------------

        end_datetime = datetime.fromtimestamp(
            end_timestamp
        )

        end_time = end_datetime.time()

        # -------------------------------------------------
        # Convert duration seconds to HH:MM:SS
        # -------------------------------------------------

        duration_seconds = int(
            max(
                0,
                duration_seconds
            )
        )

        hours = int(
            duration_seconds // 3600
        )

        minutes = int(
            (duration_seconds % 3600) // 60
        )

        seconds = int(
            duration_seconds % 60
        )

        # -------------------------------------------------
        # MySQL TIME value
        # -------------------------------------------------

        if hours >= 24:

            # MySQL TIME supports > 24 hours.
            duration = (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        else:

            duration = time(
                hour=hours,
                minute=minutes,
                second=seconds
            )

        # -------------------------------------------------
        # UPDATE QUERY
        # -------------------------------------------------

        query = """
            UPDATE tbl_material_violations

            SET
                end_time = %s,
                duration = %s

            WHERE id = %s
        """

        values = (
            end_time,
            duration,
            violation_id
        )

        cursor.execute(
            query,
            values
        )

        connection.commit()

        # -------------------------------------------------
        # CHECK ROW
        # -------------------------------------------------

        if cursor.rowcount == 0:

            print(
                f"[DATABASE] No violation found "
                f"with ID {violation_id}."
            )

            return False

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        print(
            f"[DATABASE] Violation {violation_id} "
            f"updated successfully."
        )

        print(
            f"[DATABASE] End: "
            f"{end_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        print(
            f"[DATABASE] Duration: "
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

        return True

    except Error as e:

        print(
            f"[DATABASE] Update error: {e}"
        )

        if connection:

            connection.rollback()

        return False

    finally:

        if cursor is not None:

            cursor.close()

        if (
            connection is not None
            and connection.is_connected()
        ):

            connection.close()

# =========================================================
# UPDATE VIOLATION VIDEO PATH
# =========================================================

def update_violation_video(
    violation_id,
    video_path
):

    if violation_id is None:

        print(
            "[DATABASE] Invalid violation ID."
        )

        return False

    if not video_path:

        print(
            "[DATABASE] Invalid video path."
        )

        return False

    connection = get_connection()

    if connection is None:

        print(
            "[DATABASE] Unable to update violation video."
        )

        return False

    cursor = None

    try:

        cursor = connection.cursor()

        # -------------------------------------------------
        # UPDATE VIDEO PATH
        # -------------------------------------------------

        query = """
            UPDATE tbl_material_violations

            SET
                violation_video = %s

            WHERE id = %s
        """

        values = (
            video_path,
            violation_id
        )

        cursor.execute(
            query,
            values
        )

        connection.commit()

        # -------------------------------------------------
        # CHECK WHETHER ROW WAS UPDATED
        # -------------------------------------------------

        if cursor.rowcount == 0:

            print(
                f"[DATABASE] No violation found "
                f"with ID {violation_id}."
            )

            return False

        print(
            f"[DATABASE] Video path updated "
            f"for violation ID {violation_id}."
        )

        print(
            f"[DATABASE] Video: {video_path}"
        )

        return True

    except Error as e:

        print(
            f"[DATABASE] Video update error: {e}"
        )

        connection.rollback()

        return False

    finally:

        if cursor is not None:

            cursor.close()

        if connection is not None and connection.is_connected():

            connection.close()        

    #     # -------------------------------------------------
    #     # CHECK WHETHER ROW WAS UPDATED
    #     # -------------------------------------------------

    #     if cursor.rowcount == 0:

    #         print(
    #             f"[DATABASE] No violation found "
    #             f"with ID {violation_id}."
    #         )

    #         return False

    #     print(
    #         f"[DATABASE] Violation {violation_id} "
    #         f"updated successfully."
    #     )

    #     print(
    #         f"[DATABASE] End: "
    #         f"{end_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    #     )

    #     print(
    #         f"[DATABASE] Duration: "
    #         f"{hours:02d}:"
    #         f"{minutes:02d}:"
    #         f"{seconds:02d}"
    #     )

    #     return True

    # except Error as e:

    #     print(
    #         f"[DATABASE] Update error: {e}"
    #     )

    #     connection.rollback()

    #     return False

    # finally:

    #     if cursor is not None:

    #         cursor.close()

    #     if connection is not None and connection.is_connected():

    #         connection.close()


        # def update_violation(violation_id, end_timestamp, duration_seconds):

        #     print("\n[DATABASE] ===============================")
        #     print("[DATABASE] UPDATE VIOLATION CALLED")
        #     print(f"[DATABASE] Violation ID   : {violation_id}")
        #     print(f"[DATABASE] End timestamp  : {end_timestamp}")
        #     print(f"[DATABASE] Duration sec   : {duration_seconds}")
        #     print("[DATABASE] ===============================")

        #     if violation_id is None:
        #         print("[DATABASE] ERROR: violation_id is None")
        #         return False

        #     connection = get_connection()

        #     if connection is None:
        #         print("[DATABASE] ERROR: Could not connect to database")
        #         return False

        #     cursor = None

        #     try:

        #         cursor = connection.cursor()

        #         end_datetime = datetime.fromtimestamp(end_timestamp)
        #         end_time = end_datetime.time()

        # # Convert seconds -> HH:MM:SS
        #         hours = int(duration_seconds // 3600)
        #         minutes = int((duration_seconds % 3600) // 60)
        #         seconds = int(duration_seconds % 60)

        #     # MySQL TIME can store durations greater than 24 hours,
        #     # but Python datetime.time cannot.
        #         if hours >= 24:
        #             duration = (
        #                 f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        #             )
        #         else:
        #             duration = time(
        #                 hour=hours,
        #                 minute=minutes,
        #                 second=seconds
        #             )

        #         print(f"[DATABASE] End time       : {end_time}")
        #         print(f"[DATABASE] Duration        : {duration}")

        #         query = """
        #             UPDATE tbl_material_violations
        #             SET
        #                 end_time = %s,
        #                 duration = %s
        #             WHERE id = %s
        #         """

        #         values = (
        #             end_time,
        #             duration,
        #             violation_id
        #         )

        #         cursor.execute(query, values)

        #         print(f"[DATABASE] Rows affected  : {cursor.rowcount}")

        #         connection.commit()

        #         if cursor.rowcount == 0:
        #             print(
        #                 f"[DATABASE] WARNING: No row found with ID {violation_id}"
        #             )
        #             return False

        #         print(
        #             f"[DATABASE] SUCCESS: Violation {violation_id} updated."
        #         )

        #         return True

        #     except Error as e:

        #         print(f"[DATABASE] UPDATE ERROR: {e}")

        #         if connection:
        #             connection.rollback()

        #         return False

        #     finally:

        #         if cursor:
        #             cursor.close()

        #         if connection:
        #             connection.close()

