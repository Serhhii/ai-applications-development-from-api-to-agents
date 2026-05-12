from enum import StrEnum
from pathlib import Path

import psycopg2
import pypdf
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


def _read_file(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8")


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"
    COSINE_DISTANCE = "cosine"


class TextProcessor:

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
        )

    def _truncate_table(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors;")
            conn.commit()
        print("Vectors table truncated.")

    def _bulk_save(self, doc_name: str, chunks: list[str], embeddings: dict[int, list[float]]):
        rows = [(doc_name, chunk, str(embeddings[i])) for i, chunk in enumerate(chunks)]
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    rows,
                )
            conn.commit()

    def process_text_file(
            self,
            file_path: str,
            chunk_size: int = 300,
            overlap: int = 40,
            dimensions: int = 384,
            truncate: bool = True,
    ):
        if truncate:
            self._truncate_table()

        text = _read_file(file_path)
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"Split into {len(chunks)} chunks. Embedding...")

        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions=dimensions)
        doc_name = Path(file_path).name

        print("Saving to DB...")
        self._bulk_save(doc_name, chunks, embeddings)
        print(f"Saved {len(chunks)} chunks to DB.")

    def search(
            self,
            search_mode: SearchMode,
            query: str,
            top_k: int = 5,
            min_score: float = 0.5,
            dimensions: int = 384,
    ) -> list[str]:
        embeddings = self.embeddings_client.get_embeddings(query, dimensions=dimensions)
        query_embedding = str(embeddings[0])

        op = "<=>" if search_mode == SearchMode.COSINE_DISTANCE else "<->"

        sql = f"""
            SELECT text, embedding {op} %s::vector AS distance
            FROM vectors
            WHERE embedding {op} %s::vector <= %s
            ORDER BY distance
            LIMIT %s;
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (query_embedding, query_embedding, min_score, top_k))
                rows = cur.fetchall()

        chunks = [row['text'] for row in rows]
        for row in rows:
            print(f"  [distance={row['distance']:.4f}] {row['text'][:80]}...")
        return chunks
