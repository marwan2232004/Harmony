#pragma once
#include <vector>

int predict_knn(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const std::vector<float>& query, int k);
