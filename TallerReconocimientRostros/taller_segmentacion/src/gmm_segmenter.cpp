#include "gmm_segmenter.h"

GMMSegmenter::GMMSegmenter(const Config& cfg) : cfg_(cfg) {
    pMOG2_ = cv::createBackgroundSubtractorMOG2(
        cfg_.history, cfg_.var_threshold, cfg_.detect_shadows);

    kernel_open_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(cfg_.morph_open_k, cfg_.morph_open_k));
    kernel_close_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(cfg_.morph_close_k, cfg_.morph_close_k));
}

cv::Mat GMMSegmenter::apply(const cv::Mat& frame) {
    cv::Mat fg_mask, clean_mask;

    // Actualizar modelo GMM y obtener mascara
    pMOG2_->apply(frame, fg_mask, -1);

    // Eliminar sombras (valor 127) dejando solo primer plano (255)
    cv::threshold(fg_mask, clean_mask, 200, 255, cv::THRESH_BINARY);

    // Operaciones morfologicas para limpiar ruido
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_OPEN, kernel_open_);
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_CLOSE, kernel_close_);

    return clean_mask;
}

std::vector<cv::Rect> GMMSegmenter::getRegions(const cv::Mat& mask) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat mask_copy = mask.clone();
    cv::findContours(mask_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> regions;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < cfg_.min_area)
            continue;
        regions.push_back(cv::boundingRect(contour));
    }
    return regions;
}

cv::Mat GMMSegmenter::getBackground() const {
    cv::Mat bg;
    pMOG2_->getBackgroundImage(bg);
    return bg;
}
