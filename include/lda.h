#pragma once
#include <vector>
#include <dlib/svm.h>

using sample_type = dlib::matrix<float, 0, 1>;
using lda_kernel = dlib::linear_kernel<sample_type>;
using lda_model = dlib::decision_function<lda_kernel>;

void train_lda(const std::string path, const std::vector<std::vector<float>>& X, const std::vector<int>& y, float c);
int predict_lda(const lda_model& model, const std::vector<float>& x);
