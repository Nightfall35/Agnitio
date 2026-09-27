#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <optional>
#include <string>
#include <vector>

// Detects the most prominent face in an image, aligns it to the standard
// ArcFace 112x112 template, and extracts its embedding via an ONNX ArcFace
// model. Model weights are NOT included in this repo — see README.md for
// where to get them.
class FaceEngine {
public:
    FaceEngine(const std::string& detector_onnx_path,
               const std::string& recognizer_onnx_path,
               float detect_score_threshold = 0.9f,
               float detect_nms_threshold = 0.3f);

    // Returns a 512-d, L2-normalized embedding for the largest face found,
    // or std::nullopt if no face is detected.
    std::optional<std::vector<float>> getEmbedding(const cv::Mat& image);

private:
    cv::Mat alignFace(const cv::Mat& image, const cv::Mat& landmarks5);

    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::dnn::Net recognizer_;
};
