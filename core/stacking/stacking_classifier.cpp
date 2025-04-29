#include <iostream>
#include "stacking_classifier.hpp"

StackingClassifier::StackingClassifier(
    std::vector<std::unique_ptr<BaseEstimator>> bases,
    std::unique_ptr<BaseEstimator> meta,
    int n_folds,
    unsigned seed)
    : bases_(std::move(bases))
    , meta_(std::move(meta))
    , K_(n_folds)
    , rng_(seed)
{
    assert(K_ >= 2);
    L_ = bases_.size();
}

void StackingClassifier::fit(const MatrixXd& X, const VectorXi& y)
{
    const int N = X.rows();
    const int D = X.cols();
    const int L = L_;

    // Precompute the folds
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng_);
    std::vector<std::vector<int>> fold_indices(K_);
    std::vector<int> fold_of(N);
    for (int i = 0; i < N; ++i)
        fold_indices[idx[i] % K_].push_back(i);

    // Meta‐features matrix Z (N × L)
    MatrixXd Z = MatrixXd::Zero(N, L);

    // Parallelize the training of base learners
    #pragma omp parallel for
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < K_; ++k) {
            const auto& test_idx = fold_indices[k];
            std::vector<int> train_idx;
            for (int k_inner = 0; k_inner < K_; ++k_inner) {
                if (k_inner != k) {
                    train_idx.insert(train_idx.end(),
                        fold_indices[k_inner].begin(), fold_indices[k_inner].end());
                }
            }

            // Train base model l on train_idx
            std::cout << "Training base model " << l + 1 << " on fold " << k + 1 << std::endl;
            MatrixXd Xtr(train_idx.size(), D);
            VectorXi ytr(train_idx.size());
            for (size_t i = 0; i < train_idx.size(); ++i) {
                Xtr.row(i) = X.row(train_idx[i]);
                ytr(i) = y(train_idx[i]);
            }
            bases_[l]->train(Xtr, ytr);

            // Predict on test_idx
            MatrixXd Xte(test_idx.size(), D);
            for (size_t i = 0; i < test_idx.size(); ++i)
                Xte.row(i) = X.row(test_idx[i]);

            VectorXi ypred(test_idx.size());
            bases_[l]->predict(Xte, ypred);

            // Write to Z without critical section (thread-safe per column l)
            for (size_t i = 0; i < test_idx.size(); ++i)
                Z(test_idx[i], l) = static_cast<double>(ypred(i));
        }
    }

    // 4) fit meta-learner on Z and y
    meta_->train(Z, y);
    // 5) Re‐train each base on FULL (X,y)
    for (int l = 0; l < L; ++l)
        bases_[l]->train(X, y);
    fitted_ = true;
}

void StackingClassifier::predict(const MatrixXd& X, VectorXi& out) const
{
	assert(fitted_);
	const int M = X.rows();
	// build meta‐features Ztest (M × L)
	MatrixXd Ztest(M, L_);
	for (int l = 0; l < L_; ++l) {
		VectorXi ypred(M);
		bases_[l]->predict(X, ypred);
		for (int i = 0; i < M; ++i)
			Ztest(i, l) = double(ypred(i));
	}
	// final
	meta_->predict(Ztest, out);
}