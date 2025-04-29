#include "../../include/knn.h"
#include "estimators.hpp"

namespace harmony
{
    SVM::SVM(double C, double gamma)
    {
        rbf_trainer.set_c(C);
        rbf_trainer.set_kernel(kernel_type(gamma));
        ovo_trainer.set_trainer(rbf_trainer);
    }

    void SVM::train(const MatrixXd &X, const VectorXi &y)
    {
        std::vector<sample_type> samples;
        std::vector<int> labels;
        samples.reserve(X.rows());
        labels.reserve(X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            samples.push_back(to_dlib_vec(X.row(i)));
            labels.push_back(y(i));
        }
        decision_function_ = ovo_trainer.train(samples, labels);
    }

    void SVM::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        y_pred.resize(X.rows());
        for (int i = 0; i < X.rows(); ++i)
        {
            auto sample = to_dlib_vec(X.row(i));
            int pred = decision_function_(sample);
            y_pred(i) = pred;
        }
    }

    ExtraTrees::ExtraTrees(std::size_t nTrees, std::size_t minLeafSize, std::size_t nClasses)
        : nClasses_(nClasses), nTrees_(nTrees), minLeafSize_(minLeafSize) {}

    void ExtraTrees::train(const MatrixXd &X, const VectorXi &y)
    {
        arma::mat data(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                data(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            labels[i] = static_cast<size_t>(y(i));
        }

        mlpack::RandomForest<> rf;
        rf = mlpack::RandomForest<>(data, labels,
                                        nClasses_,
                                        nTrees_,
                                        minLeafSize_,
                                        0,     // minimum gain split
                                        0,     // maximum depth
                                        1); 

        model_ = std::move(rf);
    }

    void ExtraTrees::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                testData(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        y_pred.resize(X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            y_pred(i) = static_cast<int>(predictions(i));
        }
    }

    RandomForest::RandomForest(std::size_t nTrees, std::size_t minLeafSize, std::size_t nClasses)
        : nClasses_(nClasses), nTrees_(nTrees), minLeafSize_(minLeafSize) {}
    
    void RandomForest::train(const MatrixXd &X, const VectorXi &y)
    {
        arma::mat data(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                data(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            labels[i] = static_cast<size_t>(y(i));
        }

        mlpack::RandomForest<> rf;
        rf = mlpack::RandomForest<>(data, labels,
                                        nClasses_,
                                        nTrees_,
                                        minLeafSize_,
                                        0,     // minimum gain split
                                        0,     // maximum depth
                                        5); 

        model_ = std::move(rf);
    }

    void RandomForest::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                testData(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        y_pred.resize(X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            y_pred(i) = static_cast<int>(predictions(i));
        }
    }

    KNN::KNN(std::size_t k, std::string metric)
        : k_(k), metric_(std::move(metric))
    {
        if (metric_ != "euclidean" && metric_ != "manhattan")
            throw std::invalid_argument("Unknown distance metric: " + metric_);
        if (k_ < 1)
            throw std::invalid_argument("Number of neighbors (k) must be at least 1");
    }

    void KNN::train(const MatrixXd &X, const VectorXi &y)
    {
        train_features_.resize(X.rows());
        for (int i = 0; i < X.rows(); ++i)
        {
            train_features_[i].resize(X.cols());
            for (int j = 0; j < X.cols(); ++j)
            {
                train_features_[i][j] = static_cast<float>(X(i, j));
            }
        }

        train_labels_.resize(y.size());
        for (int i = 0; i < y.size(); ++i)
        {
            train_labels_[i] = y(i);
        }
    }

    void KNN::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        y_pred.resize(X.rows());

        for (int i = 0; i < X.rows(); ++i)
        {
            std::vector<float> query(X.cols());
            for (int j = 0; j < X.cols(); ++j)
            {
                query[j] = static_cast<float>(X(i, j));
            }

            y_pred(i) = predict_knn(train_features_, train_labels_, query, k_, metric_);
        }
    }

    LR::LR(double lambda, std::size_t nClasses)
        : lambda_(lambda), nClasses_(nClasses)
    {}

    void LR::train(const MatrixXd &X, const VectorXi &y)
    {
        const int n = X.rows(), d = X.cols();
        
        arma::mat trainData(d, n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < d; j++)
                trainData(j, i) = X(i, j);
        
        arma::Row<size_t> labels(n);
        for (size_t i = 0; i < n; ++i)
            labels(i) = static_cast<size_t>(y(i));
        
        model_ = mlpack::SoftmaxRegression<>(trainData,
            labels,
            nClasses_,
            lambda_);
    }

    void LR::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        const int n = X.rows(), d = X.cols();

        arma::mat testData(d, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
            testData(j, i) = X(i, j);

        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        y_pred.resize(n);
        for (size_t i = 0; i < n; ++i)
            y_pred(i) = static_cast<int>(predictions(i));
    }
}