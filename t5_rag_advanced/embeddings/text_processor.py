from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor
from pypdf import PdfReader

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self, cursor):
        """Clean up the vectors table"""
        cursor.execute("TRUNCATE TABLE vectors")

    def _save_chunk(self, cursor, document_name, text, embedding):
        """Store a single text chunk with its embedding"""
        cursor.execute(
            "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
            (document_name, text, str(embedding))
        )

    def _read_text(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def process_text_file(self, file_path: str, chunk_size: int, overlap: int, dimensions: int, truncate: bool = False):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if truncate:
                    self._truncate_table(cur)

                text = self._read_text(file_path)
                chunks = chunk_text(text, chunk_size, overlap)
                embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

                for idx, chunk in enumerate(chunks):
                    embedding = embeddings[idx]
                    self._save_chunk(cur, file_path, chunk, embedding)
            conn.commit()

    def search(self, search_mode: SearchMode, query: str, top_k: int, min_score: float, dimensions: int):
        query_embedding = self.embeddings_client.get_embeddings([query], dimensions)[0]

        operator = "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"

        sql = f"""
        SELECT text, embedding {operator} %s::vector AS distance
        FROM vectors
        WHERE embedding {operator} %s::vector <= %s
        ORDER BY distance
        LIMIT %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (str(query_embedding), str(query_embedding), min_score, top_k))
                results = cur.fetchall()

        return [row['text'] for row in results]


# SELECT text, embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector AS distance
# FROM vectors
# WHERE embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector <= {score}
# ORDER BY distance
# LIMIT {top_k};
