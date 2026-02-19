#include "Tensor.h"
#include "../core/autograd/GradFn.h"

namespace ctorch {

Tensor::Tensor(const std::vector<std::vector<double>>& data,
               bool requires_grad,
               const std::string& device)
    : data(data),
      requires_grad(requires_grad),
      device(device),
      grad_fn(nullptr)
{
    shape = {static_cast<int>(data.size()),
             static_cast<int>(data.empty() ? 0 : data[0].size())};

    grad.resize(shape[0], std::vector<double>(shape[1], 0.0));
}

void Tensor::backward() {
    if (!requires_grad)
        return;

    // Initialize gradient of final tensor to 1
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            grad[i][j] = 1.0;

    if (grad_fn) {
        grad_fn->backward(shared_from_this());
    }
}

std::vector<std::vector<double>> Tensor::T() const {
    std::vector<std::vector<double>> transposed(cols(),
        std::vector<double>(rows()));

    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            transposed[j][i] = data[i][j];

    return transposed;
}

int Tensor::rows() const {
    return shape[0];
}

int Tensor::cols() const {
    return shape[1];
}

const std::vector<std::vector<double>>& Tensor::get_data() const {
    return data;
}

const std::vector<std::vector<double>>& Tensor::get_grad() const {
    return grad;
}

bool Tensor::requires_grad_enabled() const {
    return requires_grad;
}

void Tensor::set_grad_fn(std::shared_ptr<GradFn> fn) {
    grad_fn = fn;
}

std::shared_ptr<GradFn> Tensor::get_grad_fn() const {
    return grad_fn;
}

void Tensor::accumulate_grad(
    const std::vector<std::vector<double>>& grad_input)
{
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            grad[i][j] += grad_input[i][j];
}

}
