#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Detecta cruces por cero en el resultado del Laplaciano
Mat zeroCrossing(const Mat& laplacian) {
    Mat result = Mat::zeros(laplacian.size(), CV_8U);
    for (int i = 1; i < laplacian.rows - 1; i++) {
        for (int j = 1; j < laplacian.cols - 1; j++) {
            short center = laplacian.at<short>(i, j);
            // Revisar vecinos opuestos (horizontal, vertical, diagonales)
            short neighbors[4][2] = {
                {laplacian.at<short>(i, j-1), laplacian.at<short>(i, j+1)},
                {laplacian.at<short>(i-1, j), laplacian.at<short>(i+1, j)},
                {laplacian.at<short>(i-1, j-1), laplacian.at<short>(i+1, j+1)},
                {laplacian.at<short>(i-1, j+1), laplacian.at<short>(i+1, j-1)}
            };
            for (auto& pair : neighbors) {
                if ((pair[0] > 0 && pair[1] < 0) || (pair[0] < 0 && pair[1] > 0)) {
                    if (abs(pair[0] - pair[1]) > 20) { // umbral para reducir ruido
                        result.at<uchar>(i, j) = 255;
                        break;
                    }
                }
            }
        }
    }
    return result;
}

int main() {
    int opcion = 0;

    do {
        destroyAllWindows();
        cout << "\n===== Taller 1 - Corte 2 =====" << endl;
        cout << "1. Camara con filtro LoG (bordes)" << endl;
        cout << "2. Camara con Zero Crossing" << endl;
        cout << "3. Camara con filtro Sobel" << endl;
        cout << "4. Camara con filtro Scharr" << endl;
        cout << "5. Camara con filtro Laplaciano" << endl;
        cout << "6. Camara con Sobel Magnitude" << endl;
        cout << "7. Camara con Transformada de Hough (lineas)" << endl;
        cout << "0. Salir" << endl;
        cout << "Opcion: ";
        cin >> opcion;

        switch (opcion) {
            case 1: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, blurred, log_result;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    // Convertir a escala de grises
                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    // Gaussian blur (la G del LoG)
                    GaussianBlur(gray, blurred, Size(5, 5), 1.0);
                    // Laplaciano (la L del LoG)
                    Laplacian(blurred, log_result, CV_16S, 3);
                    convertScaleAbs(log_result, log_result);

                    imshow("Original", frame);
                    imshow("LoG - Bordes", log_result);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 2: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, blurred, lap, zc;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    GaussianBlur(gray, blurred, Size(5, 5), 1.0);
                    Laplacian(blurred, lap, CV_16S, 3);
                    zc = zeroCrossing(lap);

                    imshow("Original", frame);
                    imshow("Zero Crossing", zc);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 3: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, sobel_x, sobel_y, sobel;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    Sobel(gray, sobel_x, CV_16S, 1, 0, 3); // Gradiente en X
                    Sobel(gray, sobel_y, CV_16S, 0, 1, 3); // Gradiente en Y
                    convertScaleAbs(sobel_x, sobel_x);
                    convertScaleAbs(sobel_y, sobel_y);
                    addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, sobel); // Magnitud combinada

                    imshow("Original", frame);
                    imshow("Sobel X", sobel_x);
                    imshow("Sobel Y", sobel_y);
                    imshow("Sobel - Bordes", sobel);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 4: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, scharr_x, scharr_y, scharr;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    Scharr(gray, scharr_x, CV_16S, 1, 0); // Gradiente en X
                    Scharr(gray, scharr_y, CV_16S, 0, 1); // Gradiente en Y
                    convertScaleAbs(scharr_x, scharr_x);
                    convertScaleAbs(scharr_y, scharr_y);
                    addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0, scharr);

                    imshow("Original", frame);
                    imshow("Scharr X", scharr_x);
                    imshow("Scharr Y", scharr_y);
                    imshow("Scharr - Bordes", scharr);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 5: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, lap_result;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    Laplacian(gray, lap_result, CV_16S, 3);
                    convertScaleAbs(lap_result, lap_result);

                    imshow("Original", frame);
                    imshow("Laplaciano", lap_result);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 6: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, sobel_x, sobel_y, magnitude;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    Sobel(gray, sobel_x, CV_64F, 1, 0, 3);
                    Sobel(gray, sobel_y, CV_64F, 0, 1, 3);
                    // Magnitud real: sqrt(Gx^2 + Gy^2)
                    Mat mag;
                    cv::magnitude(sobel_x, sobel_y, mag);
                    normalize(mag, mag, 0, 255, NORM_MINMAX);
                    mag.convertTo(magnitude, CV_8U);

                    imshow("Original", frame);
                    imshow("Sobel Magnitude", magnitude);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 7: {
                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cout << "Error: no se pudo abrir la camara" << endl;
                    break;
                }

                Mat frame, gray, edges, hough_result;
                vector<Vec2f> lines;
                cout << "Presiona ESC para volver al menu" << endl;

                while (true) {
                    cap >> frame;
                    if (frame.empty()) break;

                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    Canny(gray, edges, 50, 150);
                    HoughLines(edges, lines, 1, CV_PI / 180, 150);

                    hough_result = frame.clone();
                    for (size_t i = 0; i < lines.size(); i++) {
                        float rho = lines[i][0], theta = lines[i][1];
                        double a = cos(theta), b = sin(theta);
                        double x0 = a * rho, y0 = b * rho;
                        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
                        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
                        line(hough_result, pt1, pt2, Scalar(0, 0, 255), 2);
                    }

                    imshow("Original", frame);
                    imshow("Canny (bordes)", edges);
                    imshow("Hough - Lineas", hough_result);

                    if (waitKey(30) == 27) break;
                }
                cap.release();
                break;
            }
            case 0:
                cout << "Saliendo..." << endl;
                break;
            default:
                cout << "Opcion no valida" << endl;
        }
    } while (opcion != 0);

    destroyAllWindows();
    return 0;
}
