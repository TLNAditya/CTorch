#include<cmath>
#include<vector>
#include "../core/Tensor.h"
using namespace std;

namespace ctorch{

class BinaryCrossEntropyLoss{
public:
    double forward(Tensor &output, Tensor &label){
        //formula:
        //bce = -1/n summation(yi(log(y^i))+(1-yi)(log(1-y^i)))
        //n = number of classes
        const double eps = 1e-9;
        int n = output.size();
        double loss = 0;
        for(int i = 0;i < n;i++){
            double y_hat = max(eps, min(1.0 - eps, output[i]));
            double y = label[i];
            loss += y * log(y_hat) + (1 - y) * log(1 - y_hat);
        }
        return -loss/n;
    }
};
}