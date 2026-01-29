#include "../core/Tensor.h"
#include <bits/stdc++.h>
using namespace std;

namespace ctorch{
class Matmul{
    public:
    Tensor compute(const Tensor &mat1,const Tensor &mat2){
        try{
            int r1 = mat1.rows();
            int c1 = mat1.cols();
            int r2 = mat2.rows();
            int c2 = mat2.cols();
            const auto& A = mat1.get_data();
            const auto& B = mat2.get_data();
            if(c1!=r2)throw invalid_argument("Matrix mul: col of mat1 & row of mat2");
            vector<vector<double>> result(r1,vector<double>(c2));
            #pragma omp parallel for
            for(int i = 0;i < r1;i++){
                for(int j = 0;j < c1; j++){
                    result[i][j] = 0;
                    for(int k = 0;k < c2;k++){
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return Tensor(result,mat1.requires_grad_enabled() || mat2.requires_grad_enabled());
        }
        catch(exception &e){
            cout << e.what() << endl;
        }
    } 
    
};
}