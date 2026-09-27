#include "db.hpp"

#include <pqxx/pqxx>
#include <sstream>

FaceDatabase::FaceDatabase(const std::string& conninfo) : conninfo_(conninfo) {
    // Fail fast at startup if the connection info is bad, rather than on
    // the first request.
    pqxx::connection conn(conninfo_);
}

std::string FaceDatabase::toVectorLiteral(const std::vector<float>& embedding) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < embedding.size(); ++i) {
        if (i > 0) oss << ",";
        oss << embedding[i];
    }
    oss << "]";
    return oss.str();
}

int64_t FaceDatabase::findOrCreatePerson(const std::string& name) {
    pqxx::connection conn(conninfo_);
    pqxx::work txn(conn);

    auto existing = txn.exec_params("SELECT id FROM people WHERE name = $1", name);
    if (!existing.empty()) {
        int64_t id = existing[0][0].as<int64_t>();
        txn.commit();
        return id;
    }

    auto inserted = txn.exec_params(
        "INSERT INTO people (name) VALUES ($1) RETURNING id", name);
    int64_t id = inserted[0][0].as<int64_t>();
    txn.commit();
    return id;
}

void FaceDatabase::addEmbedding(int64_t person_id,
                                 const std::string& image_path,
                                 const std::vector<float>& embedding,
                                 const std::string& model_name) {
    pqxx::connection conn(conninfo_);
    pqxx::work txn(conn);

    txn.exec_params(
        "INSERT INTO face_embeddings (person_id, image_path, embedding, model_name) "
        "VALUES ($1, $2, $3, $4)",
        person_id, image_path, toVectorLiteral(embedding), model_name);

    txn.commit();
}

std::vector<FaceMatch> FaceDatabase::searchSimilarFaces(const std::vector<float>& embedding,
                                                         int top_k,
                                                         double match_threshold,
                                                         const std::string& model_name) {
    pqxx::connection conn(conninfo_);
    pqxx::work txn(conn);

    std::string vec_literal = toVectorLiteral(embedding);

    auto rows = txn.exec_params(
        "SELECT p.id, p.name, fe.image_path, 1 - (fe.embedding <=> $1) AS similarity "
        "FROM face_embeddings fe "
        "JOIN people p ON p.id = fe.person_id "
        "WHERE fe.model_name = $2 "
        "ORDER BY fe.embedding <=> $1 "
        "LIMIT $3",
        vec_literal, model_name, top_k);

    txn.commit();

    std::vector<FaceMatch> results;
    results.reserve(rows.size());
    for (const auto& row : rows) {
        FaceMatch match;
        match.person_id = row[0].as<int64_t>();
        match.name = row[1].as<std::string>();
        match.image_path = row[2].as<std::string>();
        match.similarity = row[3].as<double>();
        match.is_match = match.similarity >= match_threshold;
        results.push_back(match);
    }
    return results;
}
