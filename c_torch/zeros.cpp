#include<bits/stdc++.h>
using namespace std;

class Zeros{

    private:
    vector<vector<double>> zero;
    vector<int> shape;
    bool requires_grad;

    public:
    Zeros(vector<int>shape,bool requires_grad=false){
        this->requires_grad = requires_grad;
        this->shape=shape;
        
    }
    vector<vector<double>> zeros(){
        try{
            if (shape.size() != 2){
                throw invalid_argument("zeros() requires 2D shape!");
            }
            if(shape[0]<=0 || shape[1] <= 0){
                throw invalid_argument("Matrix dimensions must be > 0");
            }
            
            zero.resize(shape[0],vector<double>(shape[1],0.0));
            return zero;  
        }
        catch(exception &e){
            cout << e.what();
        }
        
    }
    string dtype() const{
        return "float64";
    }
};

