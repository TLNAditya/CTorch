#include "../core/Tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace std;
namespace ctorch {

class Softmax {
public:
    Tensor forward(const Tensor& input) {
        int r = input.rows();
        int c = input.cols();

        std::vector<std::vector<double>> out(r, std::vector<double>(c));
        
        auto data = input.get_data(); 

        //computing the denominator part first because we need to divide it for every element.
        double denominator = 0.0;
        double max_element = INT_MIN;
        for(int i = 0;i < r;i++){
            for(int j = 0;j < c;j++){
                if(data[i][j]>max_element){
                    max_element = max(max_element,data[i][j]);
                }
            }
        }
        
        //Softmax formula s(x): (e^x)/Σ(e^xi)
        //e = 2.718

        for(int i = 0;i < r;i++){
            for(int j = 0;j < c;j++){
                denominator += exp(data[i][j] - max_element);
            }
        }  
        
        #pragma omp parallel for
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double numerator = pow(2.718,data[i][j]);
                out[i][j] = numerator/denominator;
            }
        }

        return Tensor(out, input.requires_grad_enabled());
    }
};

}
