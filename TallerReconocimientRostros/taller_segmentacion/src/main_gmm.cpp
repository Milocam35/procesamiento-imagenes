#include "gmm_segmenter.h"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: no se pudo abrir la camara" << std::endl;
        return -1;
    }

    GMMSegmenter::Config cfg;
    GMMSegmenter segmenter(cfg);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Obtener mascara binaria limpia
        cv::Mat mask = segmenter.apply(frame);

        // Dibujar bounding boxes sobre el frame
        std::vector<cv::Rect> regions = segmenter.getRegions(mask);
        for (const auto& box : regions) {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }

        // Obtener fondo reconstruido
        cv::Mat bg = segmenter.getBackground();

        cv::imshow("Frame Original", frame);
        cv::imshow("Mascara GMM", mask);
        if (!bg.empty())
            cv::imshow("Fondo Estimado", bg);

        // ESC para salir
        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
