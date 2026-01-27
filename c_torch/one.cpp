#include <bits/stdc++.h>
#include <type_traits>
using namespace std;

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64
};

DType parse_dtype(const string& s) {
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "int32")   return DType::Int32;
    if (s == "int64")   return DType::Int64;
    throw runtime_error("Unknown dtype");
}

class TensorBase {
public:
    bool requires_grad = false;

    virtual ~TensorBase() = default;
    virtual void* data() = 0;
    virtual const vector<size_t>& shape() const = 0;
    virtual DType dtype() const = 0;
};

template <typename T>
class Tensor : public TensorBase {
private:
    vector<size_t> _shape;
    vector<T> _data;

public:
    Tensor(const vector<size_t>& shape, bool requires_grad = false) {
        this->requires_grad = requires_grad;
        _shape = shape;

        size_t total = 1;
        for (auto s : shape) total *= s;

        _data.resize(total, static_cast<T>(1));
    }

    void* data() override {
        return _data.data();
    }

    const vector<size_t>& shape() const override {
        return _shape;
    }

    DType dtype() const override {
        if (std::is_same<T, float>::value)  return DType::Float32;
        if (std::is_same<T, double>::value) return DType::Float64;
        if (std::is_same<T, int>::value)    return DType::Int32;
        if (std::is_same<T, long>::value)   return DType::Int64;
        throw runtime_error("Unknown dtype");
    }
};

unique_ptr<TensorBase> ones(
    const vector<size_t>& shape,
    DType dtype,
    bool requires_grad = false
) {
    switch (dtype) {
        case DType::Float32:
            return make_unique<Tensor<float>>(shape, requires_grad);
        case DType::Float64:
            return make_unique<Tensor<double>>(shape, requires_grad);
        case DType::Int32:
            return make_unique<Tensor<int>>(shape, requires_grad);
        case DType::Int64:
            return make_unique<Tensor<long>>(shape, requires_grad);
    }
    throw runtime_error("Unsupported dtype");
}


