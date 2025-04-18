#pragma once
#include <vector>
#include <dlib/svm.h>

using sample_type = dlib::matrix<float, 0, 1>;
using rbf_kernel = dlib::radial_basis_kernel<sample_type>;
using decision_func = dlib::decision_function<rbf_kernel>;

void train_svm_rbf(
    const std::string path,
    const std::vector<std::vector<float>>& X,
    const std::vector<int>& y,
    float C,
    float gamma
);

int predict_svm_rbf(const decision_func& df, const std::vector<float>& x);
