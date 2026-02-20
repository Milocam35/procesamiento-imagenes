#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>

class GMMSegmenter {
public:
    struct Config {
        int history = 500;            // Cuadros para construir modelo
        double var_threshold = 16.0;  // Umbral distancia Mahalanobis^2
        bool detect_shadows = true;   // Detectar sombras (gris=127)
        int min_area = 1500;          // Area minima de contorno valido
        int morph_open_k = 3;         // Tamano kernel apertura morfologica
        int morph_close_k = 7;        // Tamano kernel cierre morfologico
    };

    explicit GMMSegmenter(const Config& cfg);

    // Aplica GMM a un cuadro, retorna mascara binaria limpia
    cv::Mat apply(const cv::Mat& frame);

    // Extrae bounding boxes validos de la mascara
    std::vector<cv::Rect> getRegions(const cv::Mat& mask) const;

    // Retorna el fondo estimado actual del modelo
    cv::Mat getBackground() const;

private:
    Config cfg_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2_;
    cv::Mat kernel_open_;
    cv::Mat kernel_close_;
};
