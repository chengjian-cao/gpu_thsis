#include <vector>

typedef std::vector<std::vector<int>> Matrix;

class MatrixOperations {
public:
    // 随机生成并相乘两个指定大小的矩阵
    static Matrix generateAndMultiplyMatrices(int sizeA, int sizeB);

    // 打印矩阵
    static void printMatrix(const Matrix &mat);
};


