import sqlite3



def get_db_connection():
    conn = sqlite3.connect('/home/zachary/PycharmProjects/SparseMate/SparseMate.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


def get_tables():
    conn = get_db_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    print(f"Found {len(tables)} tables in the database.")
    return [table['name'] for table in tables]



def get_features(table_name):
    conn = get_db_connection()
    query = f"SELECT DISTINCT feature FROM {table_name} ORDER BY feature"
    features = conn.execute(query).fetchall()
    conn.close()
    return [row['feature'] for row in features]
