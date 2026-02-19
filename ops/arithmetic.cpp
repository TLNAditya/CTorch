#include "../core/Tensor.h"
#include<bits/stdc++.h>
using namespace std;

namespace ctorch{
    shared_ptr<Node> mul(shared_ptr<Node> a, shared_ptr<Node> b) {
        auto out = make_shared<Node>(a->data * b->data,
                                    vector{a, b},
                                    "*");

        out->backward_fn = [a, b, out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };

        return out;
    }
    shared_ptr<Node> mul(shared_ptr<Node> a, shared_ptr<Node> b) {
        auto out = make_shared<Node>(a->data * b->data,
                                    vector{a, b},
                                    "*");

        out->backward_fn = [a, b, out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };

        return out;
    }



}
