#include "../core/Tensor.h" 
#include<bits/stdc++.h>
using namespace std;


namespace ctorch{
    class Mul{
        public:
        Tensor compute(Tensor &a,Tensor &b){
            int r1 = a.rows();
            int c1 = a.cols();
            int r2 = b.rows();
            int c2 = b.cols();
            if((r1!=r2)||(c1!=c2))throw invalid_argument("The size of tensor 1 and tensor 2 must be same. ");
            auto x = a.get_data();
            auto y = b.get_data();
            vector<vector<double>> result(r1,vector<double>(c1));

            #pragma omp parallel for
            for(int i = 0;i < r1;i++){
                for(int j = 0;j < c1;j++){
                    result[i][j] = x[i][j] * y[i][j];
                }
            }
            return Tensor(result,a.requires_grad_enabled() || b.requires_grad_enabled());
        }
    };
}