#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FaceMatch {
    int64_t person_id;
    std::string name;
    std::string image_path;
    double similarity;
    bool is_match;
};

// Thin wrapper around libpqxx for the two operations this service needs:
// storing embeddings and searching them by exact cosine distance.
class FaceDatabase {
public:
    explicit FaceDatabase(const std::string& conninfo);

    // Looks up a person by name, creating them if they don't exist yet.
    int64_t findOrCreatePerson(const std::string& name);

    void addEmbedding(int64_t person_id,
                       const std::string& image_path,
                       const std::vector<float>& embedding,
                       const std::string& model_name = "ArcFace");

    // Exact (brute-force) cosine similarity search — no approximate index —
    // ordered best match first.
    std::vector<FaceMatch> searchSimilarFaces(const std::vector<float>& embedding,
                                               int top_k,
                                               double match_threshold,
                                               const std::string& model_name = "ArcFace");

private:
    std::string conninfo_;
    static std::string toVectorLiteral(const std::vector<float>& embedding);
};
