#include "../core/Tensor.h"
#include<bits/stdc++.h>
using namespace std;

namespace ctorch{
class Add{
    public:
    Tensor compute(const Tensor &input1,const Tensor &input1){
        try{
            
            int r1 = input1.rows();
            int r2 = input2.rows();
            int c1 = input1.cols();
            int c2 = input2.cols();
            auto input1 = input1.get_data();
            auto input2 = input2.get_data();
            vector<vector<double>> result(r1,vector<double>(c1));
            if((r1 != r2) || (r2 != c2)) throw invalid_argument("Addition of error: rows and cols of two matrices should be same!!");

            #pragma omp parallel for
            for(int i = 0;i < r1;i++){
                for(int j = 0;j < c2;j++){
                    result[i][j] = input1[i][j] + input2[i][j];
                }
            }
            return Tensor(result);
        }
        catch(exception &e){
            cout << e.what() <<endl;
        }
        
    }
};

}
