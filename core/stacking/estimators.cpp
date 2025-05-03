#include "../../include/knn.h"
#include "estimators.hpp"


namespace harmony
{
    //--------------------------------------------------------------------------------------
    //-----------------------------SVM Classifier-------------------------------------------
    //--------------------------------------------------------------------------------------
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

    void SVM::predict(const MatrixXd &X, VectorXi &y_pred)
    {
        y_pred.resize(X.rows());
        for (int i = 0; i < X.rows(); ++i)
        {
            auto sample = to_dlib_vec(X.row(i));
            int pred = decision_function_(sample);
            y_pred(i) = pred;
        }
    }

    bool SVM::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/SVM_model.dat";
            dlib::serialize(filepath) << decision_function_;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving SVM model: " << e.what() << std::endl;
            return false;
        }
    }

    bool SVM::load(const std::string &directory)
    {
        try
        {
            std::string filepath = directory + "/SVM_model.dat";
            dlib::deserialize(filepath) >> decision_function_;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading SVM model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------Extra Trees Classifier-----------------------------------
    //--------------------------------------------------------------------------------------
    ExtraTrees::ExtraTrees(std::size_t nTrees, std::size_t minLeafSize, std::size_t nClasses)
        : nClasses_(nClasses), nTrees_(nTrees), minLeafSize_(minLeafSize) {}

    void ExtraTrees::train(const MatrixXd &X, const VectorXi &y)
    {
        arma::mat data(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                data(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i)
        {
            labels[i] = static_cast<size_t>(y(i));
        }

        mlpack::RandomForest<> rf;
        rf = mlpack::RandomForest<>(data, labels,
                                    nClasses_,
                                    nTrees_,
                                    minLeafSize_,
                                    0, // minimum gain split
                                    0, // maximum depth
                                    1);

        model_ = std::move(rf);
    }

    void ExtraTrees::predict(const MatrixXd &X, VectorXi &y_pred)
    {
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                testData(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        y_pred.resize(X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            y_pred(i) = static_cast<int>(predictions(i));
        }
    }

    bool ExtraTrees::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/ExtraTrees_model.bin";
            mlpack::data::Save(filepath, "Extra Tree", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving Extra Trees model: " << e.what() << std::endl;
            return false;
        }
    }

    bool ExtraTrees::load(const std::string &directory)
    {
        try
        {
            std::string filepath = directory + "/ExtraTrees_model.bin";
            mlpack::data::Load(filepath, "Extra Tree", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading Extra Trees model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------Random Forest Classifier---------------------------------
    //--------------------------------------------------------------------------------------
    RandomForest::RandomForest(std::size_t nTrees, std::size_t minLeafSize, std::size_t nClasses)
        : nClasses_(nClasses), nTrees_(nTrees), minLeafSize_(minLeafSize) {}

    void RandomForest::train(const MatrixXd &X, const VectorXi &y)
    {
        arma::mat data(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                data(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i)
        {
            labels[i] = static_cast<size_t>(y(i));
        }

        mlpack::RandomForest<> rf;
        rf = mlpack::RandomForest<>(data, labels,
                                    nClasses_,
                                    nTrees_,
                                    minLeafSize_,
                                    0, // minimum gain split
                                    0, // maximum depth
                                    5);

        model_ = std::move(rf);
    }

    void RandomForest::predict(const MatrixXd &X, VectorXi &y_pred)
    {
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                testData(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        y_pred.resize(X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            y_pred(i) = static_cast<int>(predictions(i));
        }
    }

    bool RandomForest::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/RandomForest_model.bin";
            mlpack::data::Save(filepath, "Random Forest", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving Random Forest model: " << e.what() << std::endl;
            return false;
        }
    }

    bool RandomForest::load(const std::string &directory)
    {
        try
        {
            std::string filepath = directory + "/RandomForest_model.bin";
            mlpack::data::Load(filepath, "Random Forest", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading Random Forest model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------KNN Classifier-------------------------------------------
    //--------------------------------------------------------------------------------------
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

    void KNN::predict(const MatrixXd &X, VectorXi &y_pred)
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

    bool KNN::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/KNN_model.bin";
            std::ofstream ofs(filepath, std::ios::binary);
            cereal::BinaryOutputArchive oarchive(ofs);
            oarchive(train_features_, train_labels_, k_, metric_);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving KNN model: " << e.what() << std::endl;
            return false;
        }
    }

    bool KNN::load(const std::string &directory)
    {
        try
        {
            std::string filepath = directory + "/KNN_model.bin";
            std::ifstream ifs(filepath, std::ios::binary);
            cereal::BinaryInputArchive iarchive(ifs);
            iarchive(train_features_, train_labels_, k_, metric_);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading KNN model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------Logistic Regression Classifier---------------------------
    //--------------------------------------------------------------------------------------
    LR::LR(double lambda, std::size_t nClasses)
        : lambda_(lambda), nClasses_(nClasses)
    {
    }

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

    void LR::predict(const MatrixXd &X, VectorXi &y_pred)
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

    bool LR::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/LR_model.bin";
            mlpack::data::Save(filepath, "LR", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving Logistic Regression model: " << e.what() << std::endl;
            return false;
        }
    }

    bool LR::load(const std::string &directory)
    {
        try
        {
            std::string filepath = directory + "/LR_model.bin";
            mlpack::data::Load(filepath, "LR", model_, true, mlpack::data::format::binary);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading Logistic Regression model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------Neural Network Classifier--------------------------------
    //--------------------------------------------------------------------------------------
    NeuralNet::NeuralNet(std::size_t hiddenUnits1, std::size_t hiddenUnits2, std::size_t nClasses)
        : hiddenUnits1_(hiddenUnits1), hiddenUnits2_(hiddenUnits2), nClasses_(nClasses), inputDim_(0)
    {
    }

    void NeuralNet::train(const MatrixXd &X, const VectorXi &y)
    {
        // Convert input to mlpack format from samples * features to features * samples
        arma::mat data(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                data(j, i) = X(i, j);
            }
        }

        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i)
        {
            labels[i] = static_cast<size_t>(y(i));
        }

        inputDim_ = X.cols();

        model_ = mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::HeInitialization>();
        model_.Add<mlpack::Linear>(hiddenUnits1_);
        model_.Add<mlpack::ReLU>();
        model_.Add<mlpack::Linear>(hiddenUnits2_);
        model_.Add<mlpack::ReLU>();
        model_.Add<mlpack::Linear>(nClasses_);
        model_.Add<mlpack::LogSoftMax>();


        // Train the model
        arma::mat oneHotLabels;
        mlpack::data::OneHotEncoding(labels, oneHotLabels);
        model_.Train(data, oneHotLabels);
    }

    void NeuralNet::predict(const MatrixXd& X, VectorXi& y_pred)
    {
        // Convert Eigen matrix (nSamples x nFeatures) to arma::mat (nFeatures x nSamples)
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                testData(j, i) = X(i, j);
            }
        }
    
        arma::mat predictionScores;
        int batchSize = 32; 
        model_.Predict(testData, predictionScores, 32);  // Each column = log probs for a sample
    
        // Extract the class with the highest log probability
        y_pred = VectorXi::Zero(predictionScores.n_cols);
        for (size_t i = 0; i < predictionScores.n_cols; ++i)
        {
            arma::uword maxIndex;
            predictionScores.col(i).max(maxIndex);  // Find index of max log-prob
            y_pred(i) = static_cast<int>(maxIndex);
        }
    }

    bool NeuralNet::save(const std::string &directory) const
    {
        try
        {
            std::string filepath = directory + "/NeuralNet_model.bin";

            // Save model parameters with data::Save
            mlpack::data::Save(filepath, "NeuralNet", model_, true, mlpack::data::format::binary);

            // Also save constructor parameters in a separate file for reconstruction
            std::ofstream paramFile(directory + "/NeuralNet_params.txt");
            if (paramFile.is_open())
            {
                paramFile << "hiddenUnits1=" << hiddenUnits1_ << std::endl;
                paramFile << "hiddenUnits2=" << hiddenUnits2_ << std::endl;
                paramFile << "nClasses=" << nClasses_ << std::endl;
                paramFile << "inputDim=" << inputDim_ << std::endl;
                paramFile.close();
                return true;
            }
            return false;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving Neural Network model: " << e.what() << std::endl;
            return false;
        }
    }

    bool NeuralNet::load(const std::string &directory)
    {
        try
        {
            // First load parameters
            std::ifstream paramFile(directory + "/NeuralNet_params.txt");
            std::string line;

            while (std::getline(paramFile, line))
            {
                if (line.find("hiddenUnits1=") == 0)
                {
                    hiddenUnits1_ = std::stoi(line.substr(13));
                }
                else if (line.find("hiddenUnits2=") == 0)
                {
                    hiddenUnits2_ = std::stoi(line.substr(13));
                }
                else if (line.find("nClasses=") == 0)
                {
                    nClasses_ = std::stoi(line.substr(9));
                }
                else if (line.find("inputDim=") == 0)
                {
                    inputDim_ = std::stoi(line.substr(9));
                }
            }

            // Recreate the model architecture
            model_ = mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::HeInitialization>();
            model_.Add<mlpack::Linear>(hiddenUnits1_);
            model_.Add<mlpack::ReLU>();
            model_.Add<mlpack::Linear>(hiddenUnits2_);
            model_.Add<mlpack::ReLU>();
            model_.Add<mlpack::Linear>(nClasses_);
            model_.Add<mlpack::LogSoftMax>();
    

            // Load the model weights
            std::string filepath = directory + "/NeuralNet_model.bin";
            mlpack::data::Load(filepath, "NeuralNet", model_, true, mlpack::data::format::binary);

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading Neural Network model: " << e.what() << std::endl;
            return false;
        }
    }

    //--------------------------------------------------------------------------------------
    //-----------------------------SVM MLpack Classifier-------------------------------------------
    //--------------------------------------------------------------------------------------
    SVM_ML::SVM_ML(double C, double gamma)
    : C_(C), gamma_(gamma), nClasses_(2)
    {
    }

    void SVM_ML::train(const MatrixXd &X, const VectorXi &y)
    {
        // Convert data to mlpack format - transpose since mlpack expects [features x samples]
        arma::mat trainData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                trainData(j, i) = X(i, j);
            }
        }

        // Convert labels
        arma::Row<size_t> labels(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            labels(i) = static_cast<size_t>(y(i));
            nClasses_ = std::max(nClasses_, static_cast<size_t>(y(i) + 1));
        }

        // Train the model
        model_ = mlpack::LinearSVM<>(trainData, labels, nClasses_, C_);
    }

    void SVM_ML::predict(const MatrixXd &X, VectorXi &y_pred)
    {
        // Convert data to mlpack format
        arma::mat testData(X.cols(), X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                testData(j, i) = X(i, j);
            }
        }

        // Predict
        arma::Row<size_t> predictions;
        model_.Classify(testData, predictions);

        // Convert predictions to output format
        y_pred.resize(X.rows());
        for (size_t i = 0; i < X.rows(); ++i) {
            y_pred(i) = static_cast<int>(predictions(i));
        }
    }

    bool SVM_ML::save(const std::string &directory) const
    {
        try {
            std::string filepath = directory + "/SVM_model.bin";
            mlpack::data::Save(filepath, "LinearSVM", model_, true, mlpack::data::format::binary);
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error saving SVM model: " << e.what() << std::endl;
            return false;
        }
    }

    bool SVM_ML::load(const std::string &directory)
    {
        try {
            std::string filepath = directory + "/SVM_model.bin";
            mlpack::data::Load(filepath, "LinearSVM", model_, true, mlpack::data::format::binary);
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error loading SVM model: " << e.what() << std::endl;
            return false;
        }
    }
}