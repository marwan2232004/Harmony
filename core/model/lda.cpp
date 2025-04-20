#include "lda.h"
#include <dlib/serialize.h>

void train_lda(const std::string path, const std::vector<std::vector<float>>& X, const std::vector<int>& y, float c) {
    std::vector<sample_type> samples;
    std::vector<float> labels;

    for (size_t i = 0; i < X.size(); ++i) {
        sample_type s(X[i].size());
        for (size_t j = 0; j < X[i].size(); ++j)
            s(j) = X[i][j];
        samples.push_back(s);
        labels.push_back(y[i]);
    }

    dlib::svm_c_trainer<lda_kernel> trainer;
    trainer.set_c(c);
    lda_model model = trainer.train(samples, labels);

    dlib::serialize(path) << model;
}

int predict_lda(const lda_model& model, const std::vector<float>& x) {
    sample_type sample(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        sample(i) = x[i];
    return model(sample) > 0 ? 1 : 0;
}
