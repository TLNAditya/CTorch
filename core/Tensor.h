#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include <vector>
#include <memory>
#include <string>

namespace ctorch {

class GradFn; // forward declaration

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> grad;
    std::vector<int> shape;

    bool requires_grad;
    std::string device;

    std::shared_ptr<GradFn> grad_fn;

public:
    Tensor(const std::vector<std::vector<double>>& data,
           bool requires_grad = false,
           const std::string& device = "cpu");

    // Backward
    void backward();

    // Utilities
    std::vector<std::vector<double>> T() const;

    int rows() const;
    int cols() const;

    const std::vector<std::vector<double>>& get_data() const;
    const std::vector<std::vector<double>>& get_grad() const;

    bool requires_grad_enabled() const;

    void set_grad_fn(std::shared_ptr<GradFn> fn);
    std::shared_ptr<GradFn> get_grad_fn() const;

    void accumulate_grad(const std::vector<std::vector<double>>& grad_input);
};

}

#endif
