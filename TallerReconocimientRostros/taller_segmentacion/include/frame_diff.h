#pragma once
#include <opencv2/opencv.hpp>

class FrameDifferencer {
public:
    // Constructor: recibe parametros de configuracion
    FrameDifferencer(int threshold_val, int min_contour_area,
                     double learning_rate = 0.0);

    // Establece el cuadro de fondo de referencia
    void setBackground(const cv::Mat& frame);

    // Procesa un cuadro y retorna mascara binaria
    cv::Mat process(const cv::Mat& frame);

    // Extrae bounding boxes de regiones activas
    std::vector<cv::Rect> getRegions(const cv::Mat& mask);

    // Actualiza el fondo con learning rate (0 = estatico)
    void updateBackground(const cv::Mat& frame);

private:
    cv::Mat background_;       // Fondo en escala de grises
    int threshold_val_;        // Umbral de binarizacion
    int min_area_;             // Area minima de contorno valido
    double learning_rate_;     // Alpha para actualizacion
    cv::Mat kernel_;           // Kernel morfologico
};
