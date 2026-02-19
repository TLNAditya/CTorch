#include "../core/Tensor.h"
#include <vector>

namespace ctorch {

class ReLU {
public:
    Tensor forward(const Tensor& input) {
        int r = input.rows();
        int c = input.cols();

        std::vector<std::vector<double>> out(r, std::vector<double>(c));

        auto data = input.get_data();   
        #pragma omp parallel for
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out[i][j] = std::max(0.0, data[i][j]);
            }
        }

        return Tensor(out, input.requires_grad_enabled());
    }
};

}
