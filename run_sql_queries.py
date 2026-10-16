
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"


def main():
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # Split the file into seven individual SQL statements.
    statements = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]

    if len(statements) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, found {len(statements)}"
        )

    expected_rows = {
        1: 30,
        2: 30,
        3: 11,
        4: 25,
        5: 550,
        7: 183,
    }

    with sqlite3.connect(DB_PATH) as connection:
        results = {}

        for number, statement in enumerate(statements, start=1):
            df = pd.read_sql_query(statement, connection)
            results[number] = df

            print(f"\n{'=' * 50}")
            print(f"QUERY {number}")
            print(f"{'=' * 50}")
            print(f"Rows returned: {len(df)}")
            print(df.head(8).to_string(index=False))

            if number in expected_rows:
                expected = expected_rows[number]
                if len(df) != expected:
                    raise AssertionError(
                        f"Query {number}: expected {expected} rows, "
                        f"received {len(df)}"
                    )

        # Verify Q2 revenue.
        q2 = results[2]
        total_revenue = round(q2["revenue"].sum(), 2)
        print(f"\nQ2 total revenue: INR {total_revenue:,.2f}")

        if total_revenue != 2998668.80:
            raise AssertionError("Q2 total revenue does not match.")

        top_row = q2.loc[q2["revenue"].idxmax()]
        print(
            "Q2 highest city/category: "
            f"{top_row['city']} / {top_row['category']} "
            f"— INR {top_row['revenue']:,.2f}"
        )

        # Verify Q5 average spend.
        q5 = results[5]
        if not q5.empty:
            average_spend = q5["average_spend"].iloc[0]
            print(f"Q5 average completed spend: INR {average_spend:,.2f}")

        # Verify Q6 city leaders and ranking ties.
        q6 = results[6]
        top_customers = (
            q6.sort_values(["city", "row_number"])
            .groupby("city", as_index=False)
            .first()
        )
        print("\nQ6 top customer in each city:")
        print(
            top_customers[
                ["city", "customer_id", "total_spend"]
            ].to_string(index=False)
        )

        tied_rows = int((q6["spend_rank"] != q6["row_number"]).sum())
        print(f"Q6 rows where RANK differs from ROW_NUMBER: {tied_rows}")

        if tied_rows != 8:
            raise AssertionError(
                f"Expected 8 ranking-tie rows, found {tied_rows}"
            )

        # Verify Q7 final cumulative revenue.
        q7 = results[7]
        final_cumulative = round(q7["cumulative_revenue"].iloc[-1], 2)
        print(f"\nQ7 days returned: {len(q7)}")
        print(f"Q7 final cumulative revenue: INR {final_cumulative:,.2f}")

        if final_cumulative != 2998668.80:
            raise AssertionError("Q7 cumulative revenue does not match.")

    print("\nALL SEVEN SQL QUERIES EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
