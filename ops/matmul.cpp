#include "../ops/matmul.h"
#include <vector>
#include <stdexcept>

namespace ctorch {

Tensor Matmul::forward(const Tensor& a, const Tensor& b) {
    int r1 = a.rows();
    int c1 = a.cols();
    int r2 = b.rows();
    int c2 = b.cols();

    if (c1 != r2) {
        throw std::invalid_argument(
            "Matmul error: columns of A must match rows of B"
        );
    }

    const auto& A = a.get_data();
    const auto& B = b.get_data();

    std::vector<std::vector<double>> result(
        r1, std::vector<double>(c2, 0.0)
    );

    #pragma omp parallel for
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return Tensor(
        result,
        a.requires_grad_enabled() || b.requires_grad_enabled()
    );
}

}
