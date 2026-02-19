#include "../core/Tensor.h"
#include <vector>
#include <cmath>
namespace ctorch {

class Sigmoid {
public:
    Tensor forward(const Tensor& input) {
        int r = input.rows();
        int c = input.cols();

        std::vector<std::vector<double>> out(r, std::vector<double>(c));
        //sigmoid formula: 1 + (1+e^-x)
        //e = 2.718
        auto data = input.get_data();   
        #pragma omp parallel for
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out[i][j] = 1 / (1 + pow(2.718,-(data[i][j])));
            }
        }

        return Tensor(out, input.requires_grad_enabled());
    }
};

}
