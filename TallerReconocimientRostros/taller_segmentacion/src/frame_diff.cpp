#include "frame_diff.h"

FrameDifferencer::FrameDifferencer(int threshold_val, int min_contour_area,
                                   double learning_rate)
    : threshold_val_(threshold_val),
      min_area_(min_contour_area),
      learning_rate_(learning_rate),
      kernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))) {}

void FrameDifferencer::setBackground(const cv::Mat& frame) {
    cv::cvtColor(frame, background_, cv::COLOR_BGR2GRAY);
}

cv::Mat FrameDifferencer::process(const cv::Mat& frame) {
    cv::Mat current_gray, diff_frame, binary_mask;

    cv::cvtColor(frame, current_gray, cv::COLOR_BGR2GRAY);
    cv::absdiff(background_, current_gray, diff_frame);
    cv::threshold(diff_frame, binary_mask, threshold_val_, 255, cv::THRESH_BINARY);

    // Operaciones morfologicas para limpiar ruido
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel_);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel_);

    return binary_mask;
}

std::vector<cv::Rect> FrameDifferencer::getRegions(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> regions;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < min_area_)
            continue;
        regions.push_back(cv::boundingRect(contour));
    }
    return regions;
}

void FrameDifferencer::updateBackground(const cv::Mat& frame) {
    if (learning_rate_ <= 0.0)
        return;

    cv::Mat current_gray;
    cv::cvtColor(frame, current_gray, cv::COLOR_BGR2GRAY);
    cv::addWeighted(background_, 1.0 - learning_rate_, current_gray, learning_rate_, 0, background_);
}
