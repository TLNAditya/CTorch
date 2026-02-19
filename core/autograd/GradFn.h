#ifndef CTORCH_GRADFN_H
#define CTORCH_GRADFN_H

#include <memory>
#include <vector>

namespace ctorch {

class Tensor; // forward declaration

class GradFn {
public:
    std::vector<std::weak_ptr<Tensor>> parents;

    virtual void backward(const std::shared_ptr<Tensor>& grad_output) = 0;

    virtual ~GradFn() = default;
};

}

#endif
