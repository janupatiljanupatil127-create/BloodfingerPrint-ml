"""
Database Utility
BloodPrint AI
"""

import sqlite3
import os

DATABASE_NAME = "history.db"


def get_connection():
    """
    Create SQLite connection.
    """

    return sqlite3.connect(DATABASE_NAME)


def create_database():
    """
    Create prediction history table.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT NOT NULL,

            random_forest TEXT NOT NULL,

            svm TEXT NOT NULL,

            knn TEXT NOT NULL,

            voting TEXT NOT NULL,

            prediction_time REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()

    print("Database Ready")


def save_prediction(
    image_name,
    random_forest,
    svm,
    knn,
    voting,
    prediction_time
):
    """
    Save prediction into database.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""

        INSERT INTO predictions(

            image_name,

            random_forest,

            svm,

            knn,

            voting,

            prediction_time

        )

        VALUES (?, ?, ?, ?, ?, ?)

    """, (

        image_name,

        random_forest,

        svm,

        knn,

        voting,

        prediction_time

    ))

    connection.commit()

    connection.close()


def get_history():
    """
    Fetch all predictions.
    """

    connection = get_connection()

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""

        SELECT *

        FROM predictions

        ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


def delete_history(record_id):
    """
    Delete one prediction.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        "DELETE FROM predictions WHERE id=?",

        (record_id,)

    )

    connection.commit()

    connection.close()
