#include<cmath>
#include<vector>
#include "../core/Tensor.h"
using namespace std;

namespace ctorch{
// we can't expect probabilities directly from past linear layer.
// so we should convert or expect raw result which is not in terms of probabilities.
// for this we have to use logits.
class CrossEntropyLoss {
private:
    Tensor cached_logits;
    Tensor cached_labels;

public:

    double forward(Tensor &logits, Tensor &labels) {

        cached_logits = logits;
        cached_labels = labels;

        int n = logits.size();  // assume 1D for now
        double max_logit = logits[0];

        // find max for stability
        for (int i = 1; i < n; i++) {
            max_logit = max(max_logit, logits[i]);
        }

        double sum_exp = 0.0;
        for (int i = 0; i < n; i++) {
            sum_exp += exp(logits[i] - max_logit);
        }

        double log_sum_exp = max_logit + log(sum_exp);

        double loss = 0.0;

        for (int i = 0; i < n; i++) {
            loss += labels[i] * (logits[i] - log_sum_exp);
        }

        return -loss;
    }
};

}