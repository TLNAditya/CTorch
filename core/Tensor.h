#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include <vector>

namespace ctorch {

class Tensor {
private:
    std::vector<std::vector<double>> data;
    std::vector<int> shape;
    bool requires_grad;

public:
    Tensor(const std::vector<std::vector<double>>& data,
           bool requires_grad = false);

    void backward();

    std::vector<std::vector<double>> T() const;

    int rows() const;
    int cols() const;
    const std::vector<std::vector<double>>& get_data() const;
    bool requires_grad_enabled() const;
};

} 

#endif
