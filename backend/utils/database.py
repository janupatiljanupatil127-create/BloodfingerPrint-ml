
import sqlite3

DATABASE = "history.db"


def create_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT,

            random_forest TEXT,

            svm TEXT,

            knn TEXT,

            voting TEXT,

            prediction_time REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


def save_prediction(
    image_name,
    rf,
    svm,
    knn,
    voting,
    prediction_time
):

    connection = sqlite3.connect(DATABASE)

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

        VALUES(?,?,?,?,?,?)

    """, (

        image_name,

        rf,

        svm,

        knn,

        voting,

        prediction_time

    ))

    connection.commit()

    connection.close()
