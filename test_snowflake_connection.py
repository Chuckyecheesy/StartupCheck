import os
import snowflake.connector


def main():
    key_path = "/Applications/startupproposal/rsa_key.p8"
    print("key exists:", os.path.exists(key_path))

    conn = snowflake.connector.connect(
        user="APP_USER",
        account="njqppdz-vv96787",
        private_key_file=key_path,
        warehouse="COMPUTE_WH",
        database="APP_DB",
        schema="APP_SCHEMA",
        role="APP_ROLE",
    )

    cur = conn.cursor()
    try:
        # Force session context even if defaults are NULL.
        cur.execute("USE DATABASE APP_DB")
        cur.execute("USE SCHEMA APP_DB.APP_SCHEMA")

        cur.execute(
            "SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), "
            "CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        )
        print("Session context:", cur.fetchone())

        # Keep this script as a connection check only.
        # Schema/table DDL can be executed separately with an admin role.
        cur.execute("SELECT CURRENT_VERSION()")
        print("Snowflake version:", cur.fetchone()[0])
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
