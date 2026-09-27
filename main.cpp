// Agnitio — Facial Recognition Service++
//
// Endpoints (same shape as the previous Python/FastAPI version):
//   GET  /health
//   POST /enroll  (form fields: name, file)
//   POST /search  (form fields: file, top_k [optional])
//
// Run:
//   ./agnitio
//
// Configure via environment variables:
//   FACE_DB_DSN         Postgres connection string
//   FACE_DETECTOR_ONNX  path to the YuNet face detector .onnx
//   FACE_RECOGNIZER_ONNX path to the ArcFace .onnx
//   FACE_STORAGE_DIR    where uploaded images are saved (default "storage")
//   FACE_SERVICE_PORT   port to listen on (default 8000)

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "db.hpp"
#include "face_engine.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr double kMatchThreshold = 0.68;

std::string getEnvOr(const char* name, const std::string& fallback) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : fallback;
}

std::string saveUpload(const httplib::MultipartFormData& file, const std::string& storage_dir) {
    static int counter = 0;

    std::string ext = ".jpg";
    auto dot = file.filename.find_last_of('.');
    if (dot != std::string::npos) {
        ext = file.filename.substr(dot);
    }

    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::string filename = std::to_string(now) + "_" + std::to_string(counter++) + ext;
    std::string path = storage_dir + "/" + filename;

    std::ofstream out(path, std::ios::binary);
    out << file.content;
    out.close();
    return path;
}

void sendError(httplib::Response& res, int status, const std::string& detail) {
    res.status = status;
    res.set_content(json{{"detail", detail}}.dump(), "application/json");
}

} // namespace

int main() {
    std::string storage_dir = getEnvOr("FACE_STORAGE_DIR", "storage");
    fs::create_directories(storage_dir);

    std::string detector_path = getEnvOr("FACE_DETECTOR_ONNX", "models/face_detection_yunet.onnx");
    std::string recognizer_path = getEnvOr("FACE_RECOGNIZER_ONNX", "models/arcface.onnx");
    std::string db_conninfo = getEnvOr(
        "FACE_DB_DSN",
        "dbname=facial_recognition user=postgres password=postgres host=localhost port=5432");

    FaceEngine engine(detector_path, recognizer_path);
    FaceDatabase db(db_conninfo);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Post("/enroll", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("file") || !req.has_param("name")) {
            sendError(res, 422, "Missing 'file' or 'name'");
            return;
        }

        std::string name = req.get_param_value("name");
        auto file = req.get_file_value("file");
        std::string image_path = saveUpload(file, storage_dir);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            fs::remove(image_path);
            sendError(res, 422, "Could not read image");
            return;
        }

        auto embedding = engine.getEmbedding(image);
        if (!embedding) {
            fs::remove(image_path);
            sendError(res, 422, "No face detected in image");
            return;
        }

        int64_t person_id = db.findOrCreatePerson(name);
        db.addEmbedding(person_id, image_path, *embedding);

        res.set_content(
            json{{"person_id", person_id}, {"name", name}, {"image_path", image_path}}.dump(),
            "application/json");
    });

    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("file")) {
            sendError(res, 422, "Missing 'file'");
            return;
        }

        int top_k = 5;
        if (req.has_param("top_k")) {
            try {
                top_k = std::stoi(req.get_param_value("top_k"));
            } catch (const std::exception&) {
                sendError(res, 422, "'top_k' must be an integer");
                return;
            }
        }

        auto file = req.get_file_value("file");
        std::string image_path = saveUpload(file, storage_dir);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            fs::remove(image_path);
            sendError(res, 422, "Could not read image");
            return;
        }

        auto embedding = engine.getEmbedding(image);
        fs::remove(image_path); // query image doesn't need to be kept

        if (!embedding) {
            sendError(res, 422, "No face detected in image");
            return;
        }

        auto matches = db.searchSimilarFaces(*embedding, top_k, kMatchThreshold);

        json results = json::array();
        for (const auto& m : matches) {
            results.push_back({{"person_id", m.person_id},
                                {"name", m.name},
                                {"image_path", m.image_path},
                                {"similarity", m.similarity},
                                {"is_match", m.is_match}});
        }

        res.set_content(json{{"results", results}, {"threshold", kMatchThreshold}}.dump(),
                         "application/json");
    });

    int port = std::stoi(getEnvOr("FACE_SERVICE_PORT", "8000"));
    std::cout << "Facial recognition service listening on port " << port << std::endl;
    svr.listen("0.0.0.0", port);

    return 0;
}
