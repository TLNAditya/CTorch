#ifndef CTORCH_MATMUL_H
#define CTORCH_MATMUL_H

#include "../core/Tensor.h"

namespace ctorch {

class Matmul {
public:
    static Tensor forward(const Tensor& a, const Tensor& b);
};

} 

#endif
