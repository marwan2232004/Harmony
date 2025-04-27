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
        std::vector<double> labels;
        samples.reserve(X.rows());
        labels.reserve(X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            samples.push_back(to_dlib_vec(X.row(i)));
            labels.push_back((double)y(i));
        }
        decision_function_ = ovo_trainer.train(samples, labels);
    }

    void SVM::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        y_pred.resize(X.rows());
        for (int i = 0; i < X.rows(); ++i)
        {
            auto sample = to_dlib_vec(X.row(i));
            unsigned long pred = decision_function_(sample);
            y_pred(i) = static_cast<int>(pred);
        }
    }

    ExtraTrees::ExtraTrees(std::size_t nTrees, std::size_t minLeafSize)
    {
        trainer.setNTrees(nTrees);
        trainer.setNodeSize(minLeafSize);
        // trainer.splitter().setSplitType(shark::RandomForestSplitter<shark::RealVector>::Random); // Random splits
        // trainer.splitter().setNumFeaturesPerNode(0); // 0 → all features (ExtraTrees behavior)
    }

    void ExtraTrees::train(const MatrixXd &X, const VectorXi &y)
    {
        shark::ClassificationDataset dataset;
        shark::Data<shark::RealVector> inputs(X.rows(), shark::RealVector(X.cols()));
        shark::Data<unsigned int> labels(y.size());

        for (size_t i = 0; i < X.rows(); ++i)
        {
            inputs.element(i) = shark::RealVector(to_shark_vec(X.row(i)));
            labels.element(i) = static_cast<unsigned int>(y(i));
        }

        dataset = shark::ClassificationDataset(inputs, labels);
        // trainer.train(model, dataset);
    }

    void ExtraTrees::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        y_pred.resize(X.rows());
        shark::Data<shark::RealVector> inputs(X.rows(), shark::RealVector(X.cols()));

        for (size_t i = 0; i < X.rows(); ++i)
        {
            inputs.element(i) = shark::RealVector(to_shark_vec(X.row(i)));
        }

        // auto predictions = model(inputs);
        // for(size_t i=0; i<predictions.size(); ++i) {
        //     y_pred(i) = static_cast<int>(predictions.element(i));
        // }
    }

    RandomForest::RandomForest(std::size_t nTrees, std::size_t minLeafSize)
    {
        trainer.setNTrees(nTrees);
        trainer.setNodeSize(minLeafSize);
    }

    void RandomForest::train(const MatrixXd &X, const VectorXi &y)
    {
        std::size_t n = X.rows(), d = X.cols();
        std::vector<shark::RealVector> inputVecs;
        std::vector<unsigned int> labelVecs;
        inputVecs.reserve(n);
        labelVecs.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            Eigen::VectorXd row = X.row(i);
            inputVecs.emplace_back(row.data(), row.data() + d);
            labelVecs.push_back(static_cast<unsigned int>(y(i)));
        }
        shark::Data<shark::RealVector> inputs = shark::createDataFromRange(inputVecs);
        shark::Data<unsigned int> labels = shark::createDataFromRange(labelVecs);

        shark::ClassificationDataset dataset(inputs, labels);
        trainer.train(model, dataset);
    }

    void RandomForest::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        std::vector<shark::RealVector> inputVecs;
        inputVecs.reserve(X.rows());
    
        for (Eigen::Index i = 0; i < X.rows(); ++i)
            inputVecs.emplace_back(X.row(i).data(), X.row(i).data() + X.cols());
    
        auto inputs = shark::createDataFromRange(inputVecs);
        auto predictions = model(inputs);
    
        y_pred = VectorXi::NullaryExpr(X.rows(), [&](Eigen::Index i) {
            return static_cast<int>(predictions.element(i));
        });
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

    LR::LR(double lambda1, double lambda2)
        : lambda_(lambda1), lambda2_(lambda2),
          trainer(lambda1, lambda2)
    {}

    void LR::train(const MatrixXd &X, const VectorXi &y)
    {
        const int n = X.rows(), d = X.cols();
        std::vector<shark::RealVector> inputVecs;
        std::vector<unsigned int> labelVecs;
        inputVecs.reserve(n);
        labelVecs.reserve(n);
    
        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd row = X.row(i);
            inputVecs.emplace_back(row.data(), row.data() + d);
            labelVecs.push_back(static_cast<unsigned int>(y(i)));
        }
    
        auto inputs = shark::createDataFromRange(inputVecs);
        auto labels = shark::createDataFromRange(labelVecs);
        shark::ClassificationDataset dataset(inputs, labels);
    
        trainer.train(model, dataset);
    }

    void LR::predict(const MatrixXd &X, VectorXi &y_pred) const
    {
        const int n = X.rows(), d = X.cols();
        std::vector<shark::RealVector> inputVecs;
        inputVecs.reserve(n);
    
        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd row = X.row(i);
            inputVecs.emplace_back(row.data(), row.data() + d);
        }
    
        auto inputs = shark::createDataFromRange(inputVecs);
        auto predictions = model(inputs);
    
        y_pred.resize(n);
        for (int i = 0; i < n; ++i)
        {
            y_pred(i) = static_cast<int>(predictions.element(i));
        }
    }
}