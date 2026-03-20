import psycopg2
from pgvector.psycopg2 import register_vector
import config

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST, 
            port=config.DB_PORT, 
            database=config.DB_NAME, 
            user=config.DB_USER, 
            password=config.DB_PASS
        )
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối DB: {e}")
        return None