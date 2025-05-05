#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

const int N = 6;
const double EPS = 1e-6;
const int MAX_ITER = 10000;
const double OMEGA = 1.25;

// Ax = b 的係數矩陣 A
double A[N][N] = {
    {4, -1,  0, -1,  0,  0},
    {-1, 4, -1, 0, -1,  0},
    {0, -1, 4, 0, 1, -1},
    {-1, 0, 0, 4, -1, -1},
    {0, -1, 0, -1, 4, -1},
    {0, 0, -1, 0, -1, 4}
};

// 常數向量 b
double b[N] = {0, -1, 9, 4, 8, 6};

// 用來計算向量之差的二範數
double norm(const vector<double>& x, const vector<double>& y) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    return sqrt(sum);
}

// Jacobi Method
void jacobi() {
    cout << "\n--- Jacobi Method ---\n";
    vector<double> x(N, 0), x_old(N, 0);
    int iter = 0;
    while (iter++ < MAX_ITER) {
        for (int i = 0; i < N; ++i) {
            double sigma = 0;
            for (int j = 0; j < N; ++j)
                if (i != j)
                    sigma += A[i][j] * x_old[j];
            x[i] = (b[i] - sigma) / A[i][i];
        }
        if (norm(x, x_old) < EPS) break;
        x_old = x;
    }
    for (int i = 0; i < N; ++i)
        cout << "x" << i + 1 << " = " << x[i] << endl;
    cout << "Iterations: " << iter << endl;
}

// Gauss-Seidel Method
void gaussSeidel() {
    cout << "\n--- Gauss-Seidel Method ---\n";
    vector<double> x(N, 0), x_old(N, 0);
    int iter = 0;
    while (iter++ < MAX_ITER) {
        x_old = x;
        for (int i = 0; i < N; ++i) {
            double sigma = 0;
            for (int j = 0; j < N; ++j)
                if (i != j)
                    sigma += A[i][j] * x[j];
            x[i] = (b[i] - sigma) / A[i][i];
        }
        if (norm(x, x_old) < EPS) break;
    }
    for (int i = 0; i < N; ++i)
        cout << "x" << i + 1 << " = " << x[i] << endl;
    cout << "Iterations: " << iter << endl;
}

// SOR Method
void sor() {
    cout << "\n--- SOR Method (ω = " << OMEGA << ") ---\n";
    vector<double> x(N, 0), x_old(N, 0);
    int iter = 0;
    while (iter++ < MAX_ITER) {
        x_old = x;
        for (int i = 0; i < N; ++i) {
            double sigma = 0;
            for (int j = 0; j < N; ++j)
                if (i != j)
                    sigma += A[i][j] * x[j];
            x[i] = (1 - OMEGA) * x[i] + (OMEGA * (b[i] - sigma)) / A[i][i];
        }
        if (norm(x, x_old) < EPS) break;
    }
    for (int i = 0; i < N; ++i)
        cout << "x" << i + 1 << " = " << x[i] << endl;
    cout << "Iterations: " << iter << endl;
}

// Conjugate Gradient Method
void conjugateGradient() {
    cout << "\n--- Conjugate Gradient Method ---\n";
    vector<double> x(N, 0);  // 初始解
    vector<double> r(N), p(N), Ap(N);

    // 初始 r = b - A * x
    for (int i = 0; i < N; ++i) {
        double Ax_i = 0;
        for (int j = 0; j < N; ++j)
            Ax_i += A[i][j] * x[j];
        r[i] = b[i] - Ax_i;
        p[i] = r[i];
    }

    double rs_old = 0;
    for (int i = 0; i < N; ++i)
        rs_old += r[i] * r[i];

    int iter = 0;
    while (iter++ < MAX_ITER) {
        // Ap = A * p
        for (int i = 0; i < N; ++i) {
            Ap[i] = 0;
            for (int j = 0; j < N; ++j)
                Ap[i] += A[i][j] * p[j];
        }

        double alpha_num = rs_old;
        double alpha_denom = 0;
        for (int i = 0; i < N; ++i)
            alpha_denom += p[i] * Ap[i];

        double alpha = alpha_num / alpha_denom;

        // 更新 x 和 r
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // 新的殘差長度
        double rs_new = 0;
        for (int i = 0; i < N; ++i)
            rs_new += r[i] * r[i];

        if (sqrt(rs_new) < EPS) break;  // ✔️ 真正穩定的收斂判斷

        // 更新方向 p
        double beta = rs_new / rs_old;
        for (int i = 0; i < N; ++i)
            p[i] = r[i] + beta * p[i];

        rs_old = rs_new;
    }

    for (int i = 0; i < N; ++i)
        cout << "x" << i + 1 << " = " << x[i] << endl;
    cout << "Iterations: " << iter << endl;
}


int main() {
    cout << fixed << setprecision(6);
    jacobi();
    gaussSeidel();
    sor();
    conjugateGradient();
    return 0;
}
