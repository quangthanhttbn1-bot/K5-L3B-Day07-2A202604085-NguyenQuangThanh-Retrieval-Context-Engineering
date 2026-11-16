"""Connection and embedding settings for the Day06 vector store export."""

# Matches starter/docker-compose.yml and starter/.env.example.
VECTOR_STORE_TYPE = "postgresql+pgvector"
VECTOR_STORE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
VECTOR_STORE_HOST = "127.0.0.1"
VECTOR_STORE_PORT = 55433
DATABASE_NAME = "day6_lab"
COLLECTION_OR_TABLE = "corpus_chunks"
INDEX_NAME = "corpus_chunks_meta_idx"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = 1024

# Strategy A: 500 chars with 50-char overlap; B/C: paragraph chunks up to 1,200 chars.
STRATEGIES = ("A", "B", "C")
