#include "Tensor.h"
#include <stdexcept>
using namespace std;
namespace ctorch {

Tensor::Tensor(const std::vector<std::vector<double>>& data,bool requires_grad): 
    data(data), requires_grad(requires_grad)
{
    if (data.empty())
        throw std::invalid_argument("Tensor data cannot be empty");

    int r = data.size();
    int c = data[0].size();

    for (auto& row : data) {
        if (row.size() != c)
            throw std::invalid_argument("Inconsistent row sizes");
    }

    shape = {r, c};
}

std::vector<std::vector<double>> Tensor::T() const {
    int r = shape[0], c = shape[1];
    std::vector<std::vector<double>> t(c, std::vector<double>(r));

    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            t[j][i] = data[i][j];

    return t;
}

int Tensor::rows() const { return shape[0]; }
int Tensor::cols() const { return shape[1]; }
const vector<vector<double>>& Tensor::get_data()const{return data;}
bool Tensor::requires_grad_enabled() const{return requires_grad;}
void Tensor::backward() {
   
}

} 
