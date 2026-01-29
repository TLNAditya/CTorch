#include "../core/Tensor.h"
#include "../ops/matmul.h"
#include<bits/stdc++.h>
using namespace std;

namespace ctorch{
    static std::vector<std::vector<double>> random_matrix(int rows, int cols)
{
    std::vector<std::vector<double>> m(rows, std::vector<double>(cols));
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i][j] = dist(gen);

    return m;
}
    class Linear{
    public:
    Linear(int in_shape,int output,requires_grad = false):
    W(random_matrix(in_features, out_features), requires_grad),
    b(random_matrix(1, out_features), requires_grad) //initializer list,as soon as some one call this W and bias should be initialised.
    {}

    Tensor forward(Tensor &x){
        return Matmul.forward(x,W)+b;
    }
    
};
}
