#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// ============================================================
// GRAHAM SCAN - Convex Hull desde cero O(n log n)
// ============================================================

// Producto cruz (Cross): determina el giro entre 3 puntos
// > 0: giro a la izquierda (antihorario)
// = 0: colineales
// < 0: giro a la derecha (horario)
double cross(Point A, Point B, Point C) {
    return (double)(B.x - A.x) * (C.y - A.y) - (double)(B.y - A.y) * (C.x - A.x);
}

// Distancia al cuadrado entre dos puntos
double dist2(Point A, Point B) {
    return (double)(A.x - B.x) * (A.x - B.x) + (double)(A.y - B.y) * (A.y - B.y);
}

// Punto pivot global para la comparacion de angulos
Point pivot;

// Comparador: ordena por angulo polar respecto al pivot
bool compararAngulo(Point A, Point B) {
    double cr = cross(pivot, A, B);
    if (cr == 0)
        return dist2(pivot, A) < dist2(pivot, B); // si colineales, el mas cercano primero
    return cr > 0; // antihorario primero
}

// Algoritmo Graham Scan
// 1. Pivot: punto con y mas baja (si empate, x mas baja)
// 2. Ordenar por angulo polar respecto al pivot
// 3. Pila: construir el poligono convexo
// 4. Giro: si cross <= 0, descartar punto previo
vector<Point> grahamScan(vector<Point> puntos) {
    int n = puntos.size();
    if (n < 3) return puntos;

    // Paso 1 - Pivot: encontrar el punto con y mas baja
    int idxPivot = 0;
    for (int i = 1; i < n; i++) {
        if (puntos[i].y > puntos[idxPivot].y ||
            (puntos[i].y == puntos[idxPivot].y && puntos[i].x < puntos[idxPivot].x)) {
            idxPivot = i;
        }
    }
    swap(puntos[0], puntos[idxPivot]);
    pivot = puntos[0];

    // Paso 2 - Ordenar por angulo polar respecto al pivot
    sort(puntos.begin() + 1, puntos.end(), compararAngulo);

    // Paso 3 - Pila (stack): construir el convex hull
    vector<Point> pila;
    pila.push_back(puntos[0]);
    pila.push_back(puntos[1]);
    pila.push_back(puntos[2]);

    // Paso 4 - Giro: evaluar cada punto
    for (int i = 3; i < n; i++) {
        // Mientras el giro sea a la derecha o alineado (cross <= 0), descartar
        while (pila.size() > 1 && cross(pila[pila.size() - 2], pila[pila.size() - 1], puntos[i]) <= 0) {
            pila.pop_back();
        }
        pila.push_back(puntos[i]);
    }

    // Paso 5 - Conv Hull completo
    return pila;
}

// ============================================================
// AREA - Formula de la Lazada (Shoelace) con determinantes
// Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
// ============================================================
double areaShoelace(const vector<Point>& hull) {
    double area = 0.0;
    int n = hull.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // Determinante de cada par de puntos consecutivos
        area += (double)hull[i].x * hull[j].y;
        area -= (double)hull[j].x * hull[i].y;
    }
    return abs(area) / 2.0;
}

// ============================================================
// PERIMETRO del Convex Hull
// Suma de distancias entre puntos consecutivos
// ============================================================
double perimetroHull(const vector<Point>& hull) {
    double perim = 0.0;
    int n = hull.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        perim += sqrt(dist2(hull[i], hull[j]));
    }
    return perim;
}

// ============================================================
// PERIMETRO con Canny (conteo de pixeles de borde)
// Flujo: Imagen -> Gauss -> Canny
// ============================================================
int perimetroCanny(const Mat& gray, const Mat& mascaraBinaria) {
    Mat grayBlur, cannyEdges;
    // Gauss Bell: suavizar
    GaussianBlur(gray, grayBlur, Size(5, 5), 1.5);
    // Canny: bordes
    Canny(grayBlur, cannyEdges, 50, 150);

    // Contar pixeles de borde que pertenecen a la mascara (dilatada)
    Mat maskDil;
    dilate(mascaraBinaria, maskDil, getStructuringElement(MORPH_RECT, Size(5, 5)));

    int count = 0;
    for (int i = 0; i < cannyEdges.rows; i++)
        for (int j = 0; j < cannyEdges.cols; j++)
            if (cannyEdges.at<uchar>(i, j) == 255 && maskDil.at<uchar>(i, j) == 255)
                count++;
    return count;
}

// ============================================================
// Clasificar figura geometrica por numero de vertices del hull
// ============================================================
string clasificarFigura(int vertices, double area, double perimetro) {
    // Circularidad: 4*pi*area / perimetro^2 (1.0 = circulo perfecto)
    double circularidad = (4.0 * CV_PI * area) / (perimetro * perimetro);

    if (circularidad > 0.85)
        return "Circulo";
    if (vertices == 3)
        return "Triangulo";
    if (vertices == 4)
        return "Rectangulo";
    if (vertices == 5)
        return "Pentagono";
    if (vertices == 6)
        return "Hexagono";
    return "Poligono(" + to_string(vertices) + ")";
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: no se pudo abrir la camara" << endl;
        return -1;
    }

    Mat frame, gray, umbral, mascaraBinaria, bordes;
    cout << "===== Deteccion de Figuras Geometricas =====" << endl;
    cout << "Flujo: Imagen -> Umbral -> Mascara Binaria -> Bordes -> Graham Scan -> Shoelace" << endl;
    cout << "ESC: salir" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // ---- COLUMNA IZQUIERDA: Preprocesamiento ----

        // 1. Imagen -> escala de grises
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 1.5);

        // 2. Umbral (Thresholding)
        threshold(gray, umbral, 127, 255, THRESH_BINARY_INV);

        // 3. Mascara Binaria (operaciones morfologicas)
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(umbral, mascaraBinaria, MORPH_CLOSE, kernel);
        morphologyEx(mascaraBinaria, mascaraBinaria, MORPH_OPEN, kernel);

        // 4. Bordes desde la mascara
        Canny(mascaraBinaria, bordes, 50, 150);

        // Extraer contornos
        vector<vector<Point>> contornos;
        findContours(mascaraBinaria.clone(), contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat resultado = frame.clone();

        for (size_t i = 0; i < contornos.size(); i++) {
            double areaContorno = contourArea(contornos[i]);
            double areaFrame = frame.rows * frame.cols;
            // Filtrar ruido (muy pequeno) y borde de la camara (muy grande)
            if (areaContorno < 500 || areaContorno > areaFrame * 0.90) continue;

            // ---- ALGORITMO GRAHAM SCAN (Convex Hull desde cero) ----
            vector<Point> hull = grahamScan(contornos[i]);

            if (hull.size() < 3) continue;

            // ---- CALCULOS FINALES ----

            // Area con Shoelace (determinantes)
            double areaHull = areaShoelace(hull);

            // Perimetro del hull
            double perimHull = perimetroHull(hull);

            // Aproximar poligono para contar vertices reales
            vector<Point> approx;
            double epsilon = 0.04 * perimHull;
            // Aproximacion Douglas-Peucker sobre el hull
            approxPolyDP(hull, approx, epsilon, true);
            int vertices = approx.size();

            // Clasificar la figura
            string nombre = clasificarFigura(vertices, areaHull, perimHull);

            // ---- DIBUJAR ----

            // Dibujar el convex hull en rojo
            for (size_t j = 0; j < hull.size(); j++) {
                line(resultado, hull[j], hull[(j + 1) % hull.size()],
                     Scalar(0, 0, 255), 2);
            }

            // Dibujar vertices del hull
            for (size_t j = 0; j < hull.size(); j++) {
                circle(resultado, hull[j], 4, Scalar(255, 0, 0), -1);
            }

            // Dibujar vertices aproximados en verde
            for (size_t j = 0; j < approx.size(); j++) {
                circle(resultado, approx[j], 6, Scalar(0, 255, 0), -1);
            }

            // Centro de masa
            Moments m = moments(contornos[i]);
            if (m.m00 == 0) continue;
            Point centro(m.m10 / m.m00, m.m01 / m.m00);

            // Mostrar informacion
            putText(resultado, nombre,
                    Point(centro.x - 30, centro.y - 30),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            putText(resultado, "A:" + to_string((int)areaHull),
                    Point(centro.x - 30, centro.y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            putText(resultado, "P:" + to_string((int)perimHull),
                    Point(centro.x - 30, centro.y + 20),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            putText(resultado, "V:" + to_string(vertices),
                    Point(centro.x - 30, centro.y + 40),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 0), 1);
        }

        // Mostrar cada paso
        imshow("1. Imagen Original", frame);
        imshow("2. Umbral", umbral);
        imshow("3. Mascara Binaria", mascaraBinaria);
        imshow("4. Bordes (Canny)", bordes);
        imshow("5. Resultado - Graham Scan", resultado);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
