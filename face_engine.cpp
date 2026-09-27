#include "face_engine.hpp"

#include <cmath>
#include <stdexcept>

namespace {

// Standard ArcFace 5-point reference template for a 112x112 aligned face
// (left eye, right eye, nose tip, left mouth corner, right mouth corner).
// This is the same template used by InsightFace's reference alignment.
const std::vector<cv::Point2f> kArcFaceTemplate = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
};

} // namespace

FaceEngine::FaceEngine(const std::string& detector_onnx_path,
                        const std::string& recognizer_onnx_path,
                        float detect_score_threshold,
                        float detect_nms_threshold) {
    // The input size passed here is a placeholder — setInputSize() is
    // called again per-image in getEmbedding() once the real size is known.
    detector_ = cv::FaceDetectorYN::create(
        detector_onnx_path, "", cv::Size(320, 320),
        detect_score_threshold, detect_nms_threshold);

    if (detector_.empty()) {
        throw std::runtime_error("Failed to load face detector ONNX model: " +
                                  detector_onnx_path);
    }

    recognizer_ = cv::dnn::readNetFromONNX(recognizer_onnx_path);
    if (recognizer_.empty()) {
        throw std::runtime_error("Failed to load recognition ONNX model: " +
                                  recognizer_onnx_path);
    }
}

cv::Mat FaceEngine::alignFace(const cv::Mat& image, const cv::Mat& landmarks5) {
    std::vector<cv::Point2f> src_pts;
    src_pts.reserve(5);
    for (int i = 0; i < 5; ++i) {
        src_pts.emplace_back(landmarks5.at<float>(0, i * 2),
                              landmarks5.at<float>(0, i * 2 + 1));
    }

    cv::Mat transform = cv::estimateAffinePartial2D(src_pts, kArcFaceTemplate);
    if (transform.empty()) {
        throw std::runtime_error("Failed to estimate face alignment transform");
    }

    cv::Mat aligned;
    cv::warpAffine(image, aligned, transform, cv::Size(112, 112));
    return aligned;
}

std::optional<std::vector<float>> FaceEngine::getEmbedding(const cv::Mat& image) {
    if (image.empty()) {
        return std::nullopt;
    }

    detector_->setInputSize(image.size());

    cv::Mat faces;
    detector_->detect(image, faces);

    if (faces.rows < 1) {
        return std::nullopt;
    }

    // Each row of `faces`: [x, y, w, h,
    //                        x_re, y_re, x_le, y_le, x_nt, y_nt,
    //                        x_rmc, y_rmc, x_lmc, y_lmc, score]
    // Pick the largest detected face by bounding-box area.
    int best_idx = 0;
    float best_area = 0.f;
    for (int i = 0; i < faces.rows; ++i) {
        float w = faces.at<float>(i, 2);
        float h = faces.at<float>(i, 3);
        float area = w * h;
        if (area > best_area) {
            best_area = area;
            best_idx = i;
        }
    }

    cv::Mat landmarks = faces.row(best_idx).colRange(4, 14).clone();
    cv::Mat aligned = alignFace(image, landmarks);

    // ArcFace preprocessing: BGR->RGB, scale to roughly [-1, 1], NCHW blob.
    cv::Mat blob = cv::dnn::blobFromImage(
        aligned, 1.0 / 127.5, cv::Size(112, 112),
        cv::Scalar(127.5, 127.5, 127.5), /*swapRB=*/true, /*crop=*/false);

    recognizer_.setInput(blob);
    cv::Mat output = recognizer_.forward();

    std::vector<float> embedding(output.begin<float>(), output.end<float>());

    // L2-normalize so cosine similarity comparisons are consistent
    // regardless of the raw model output's scale.
    float norm = 0.f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.f) {
        for (float& v : embedding) v /= norm;
    }

    return embedding;
}
