#include "MatrixOperations.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// 打印矩阵的实现
void MatrixOperations::printMatrix(const Matrix &mat) {
    for (const auto &row : mat) {
        for (const auto &elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}

// 生成随机矩阵
Matrix generateRandomMatrix(int rows, int cols) {
    Matrix mat(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = rand() % 10; // 生成0到9之间的随机数
        }
    }
    return mat;
}

// 生成并相乘指定大小的矩阵
Matrix MatrixOperations::generateAndMultiplyMatrices(int sizeA, int sizeB) {
    srand(static_cast<unsigned>(time(0))); // 设置随机种子

    // 生成矩阵A (sizeA x sizeB) 和矩阵B (sizeB x sizeA)
    Matrix A = generateRandomMatrix(sizeA, sizeB);
    Matrix B = generateRandomMatrix(sizeB, sizeA);

    // 初始化结果矩阵C (sizeA x sizeA)
    Matrix C(sizeA, vector<int>(sizeA, 0));

    // 矩阵相乘
    for (int i = 0; i < sizeA; ++i) {
        for (int j = 0; j < sizeA; ++j) {
            for (int k = 0; k < sizeB; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
