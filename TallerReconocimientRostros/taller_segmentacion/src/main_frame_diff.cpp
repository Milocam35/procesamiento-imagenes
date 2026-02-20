#include "frame_diff.h"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: no se pudo abrir la camara" << std::endl;
        return -1;
    }

    // Capturar primer cuadro como fondo
    cv::Mat first_frame;
    cap >> first_frame;
    if (first_frame.empty()) {
        std::cerr << "Error: no se pudo capturar el primer cuadro" << std::endl;
        return -1;
    }

    FrameDifferencer detector(30, 500, 0.0);
    detector.setBackground(first_frame);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Calcular mascara binaria
        cv::Mat mask = detector.process(frame);

        // Obtener regiones y dibujar cajas
        std::vector<cv::Rect> regions = detector.getRegions(mask);
        for (const auto& box : regions) {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }

        // Actualizar fondo (si learning_rate > 0)
        detector.updateBackground(frame);

        cv::imshow("Frame Original", frame);
        cv::imshow("Mascara Binaria", mask);

        // ESC para salir
        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
