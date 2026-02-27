#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

Mat crearKernelGaussiano(int tamaño, double sigma) {
    Mat kernel(tamaño, tamaño, CV_64F);
    int centro = tamaño / 2;
    double suma = 0.0;

    for (int i = 0; i < tamaño; i++) {
        for (int j = 0; j < tamaño; j++) {
            int x = i - centro;
            int y = j - centro;
            double valor = (1.0 / (2 * CV_PI * sigma * sigma)) *
                           exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(i, j) = valor;
            suma += valor;
        }
    }

    kernel = kernel / suma;
    return kernel;
}

void mostrarInfo(const Mat& img) {
    cout << "Filas: " << img.rows << endl;
    cout << "Columnas: " << img.cols << endl;
    cout << "Canales: " << img.channels() << endl;
    cout << "Profundidad: " << img.depth() << endl;

    Scalar media, desviacion;
    meanStdDev(img, media, desviacion);
    cout << "Media: " << media[0] << endl;
    cout << "Desviacion: " << desviacion[0] << endl;
}

void segmentacionGaussiana(const Mat& img) {
    Scalar media, desviacion;
    meanStdDev(img, media, desviacion);

    Mat mascara = Mat::zeros(img.size(), CV_8U);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            uchar pixel = img.at<uchar>(i, j);
            if (pixel > media[0] - 2 * desviacion[0] &&
                pixel < media[0] + 2 * desviacion[0]) {
                mascara.at<uchar>(i, j) = 255;
            }
        }
    }

    imshow("Segmentacion Gaussiana", mascara);
    waitKey(0);
}

void suavizadoGaussiano(const Mat& img) {
    Mat suavizada;
    GaussianBlur(img, suavizada, Size(7, 7), 1.5);
    imshow("Suavizada", suavizada);

    Scalar media, desviacion, media2, desv2;
    meanStdDev(img, media, desviacion);
    meanStdDev(suavizada, media2, desv2);
    cout << "Desviacion original: " << desviacion[0] << endl;
    cout << "Desviacion suavizada: " << desv2[0] << endl;
    waitKey(0);
}

void kernelGaussiano5x5(const Mat& img) {
    Mat kernel = crearKernelGaussiano(5, 1.0);
    cout << "Kernel Gaussiano 5x5:" << endl << kernel << endl;

    Mat suavizada;
    filter2D(img, suavizada, -1, kernel);
    imshow("Imagen Original", img);
    imshow("Suavizada (kernel 5x5 manual)", suavizada);
    waitKey(0);
}

void filtroHighpass(const Mat& img) {
    Mat laplaciano;
    Laplacian(img, laplaciano, CV_16S, 3);
    convertScaleAbs(laplaciano, laplaciano);
    imshow("Highpass - Laplaciano", laplaciano);

    Mat sharpened;
    add(img, laplaciano, sharpened);
    imshow("Imagen Afilada", sharpened);
    waitKey(0);
}

void pipelineCompleto(const Mat& img) {
    Mat gauss;
    GaussianBlur(img, gauss, Size(7, 7), 1.5);

    Mat lap;
    Laplacian(gauss, lap, CV_16S, 3);
    convertScaleAbs(lap, lap);

    Mat final_img;
    add(gauss, lap, final_img);

    imshow("Pipeline - Suavizada", gauss);
    imshow("Pipeline - Laplaciano", lap);
    imshow("Pipeline Completo", final_img);
    waitKey(0);
}

int main() {
    Mat img = imread("../data/imagen_sat.png", IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Error cargando imagen" << endl;
        return -1;
    }

    int opcion = 0;

    do {
        destroyAllWindows();
        cout << "\n===== Taller Imagen Satelital =====" << endl;
        cout << "1. Mostrar imagen original e info" << endl;
        cout << "2. Segmentacion gaussiana" << endl;
        cout << "3. Suavizado gaussiano (GaussianBlur)" << endl;
        cout << "4. Kernel gaussiano 5x5 manual" << endl;
        cout << "5. Filtro highpass (Laplaciano) + afilado" << endl;
        cout << "6. Pipeline completo (suavizado + highpass)" << endl;
        cout << "0. Salir" << endl;
        cout << "Opcion: ";
        cin >> opcion;

        switch (opcion) {
            case 1:
                imshow("Imagen Original", img);
                mostrarInfo(img);
                waitKey(0);
                break;
            case 2:
                segmentacionGaussiana(img);
                break;
            case 3:
                suavizadoGaussiano(img);
                break;
            case 4:
                kernelGaussiano5x5(img);
                break;
            case 5:
                filtroHighpass(img);
                break;
            case 6:
                pipelineCompleto(img);
                break;
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
