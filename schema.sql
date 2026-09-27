-- Facial Recognition System — PostgreSQL schema (pgvector)
-- Run this once against your database:
--   psql -d facial_recognition -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS people (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    image_path  TEXT NOT NULL,
    -- ArcFace = 512-dim. Update this if you swap in a different model.
    embedding   VECTOR(512) NOT NULL,
    model_name  TEXT NOT NULL DEFAULT 'ArcFace',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
    ON face_embeddings (person_id);

-- Deliberately NO ivfflat/hnsw approximate index on `embedding` — exact
-- cosine-distance search is used throughout to keep match accuracy at
-- its ceiling. Fine up to roughly tens of thousands of embeddings.
