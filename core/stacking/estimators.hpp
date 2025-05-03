#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <dlib/svm_threaded.h>
#include <dlib/random_forest.h>
#include <dlib/statistics.h>
#include <dlib/matrix.h>
#include <dlib/serialize.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/Csv.h>
#include <eigen3/Eigen/Dense>
#include "stacking_classifier.hpp"
#include <cereal/archives/binary.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXi;

/**
 * @brief Converts Eigen VectorXd to dlib vector format
 * @param v Eigen vector to convert
 * @return dlib::matrix<double,0,1> converted vector
 */
inline dlib::matrix<double, 0, 1> to_dlib_vec(const Eigen::VectorXd &v)
{
	dlib::matrix<double, 0, 1> m(v.size());
	for (int i = 0; i < v.size(); ++i)
		m(i) = v[i];
	return m;
}

/**
 * @brief Converts dlib vector to Eigen VectorXd format
 * @param m dlib vector to convert
 * @return Eigen::VectorXd converted vector
 */
inline Eigen::VectorXd from_dlib_vec(const dlib::matrix<double, 0, 1> &m)
{
	Eigen::VectorXd v(m.size());
	for (int i = 0; i < m.size(); ++i)
		v[i] = m(i);
	return v;
}

/**
 * @brief Converts Eigen::VectorXd to shark::RealVector
 * @param v Eigen vector to convert
 * @return shark::RealVector converted vector
 */
inline shark::RealVector to_shark_vec(const Eigen::VectorXd& v) {
    shark::RealVector vec(v.size());
    for(int i = 0; i < v.size(); ++i) {
        vec(i) = v(i);
    }
    return vec;
}

namespace harmony
{

	/**
	 * @brief Support Vector Machine classifier using dlib with RBF kernel
	 * Implements multiclass classification using one-vs-one strategy
	 */
	struct SVM : BaseEstimator
	{
		using sample_type = dlib::matrix<double, 0, 1>;
		using kernel_type = dlib::radial_basis_kernel<sample_type>;
		using ovo_trainer_type = dlib::one_vs_all_trainer<dlib::any_trainer<sample_type>, int>;;
		using df_type = typename ovo_trainer_type::trained_function_type;
		
		// trainers
		dlib::svm_c_trainer<kernel_type> rbf_trainer;
		ovo_trainer_type ovo_trainer;

		/**
		 * @brief Constructs SVM with specified parameters
		 * @param C Regularization parameter
		 * @param gamma Kernel parameter for RBF
		 */
		SVM(double C = 1.0, double gamma = 0.01);

		/**
		 * @brief Trains the SVM model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;


	private:
		df_type decision_function_;
	};

	/**
	 * @brief Extra-Trees classifier implementation using Shark
	 * Note: Extremely Randomized Trees variant
	 */
	struct ExtraTrees : BaseEstimator
	{
		/**
		 * @brief Constructs Extra-Trees classifier
		 * @param nTrees Number of trees in the forest
		 * @param minLeafSize Minimum samples required in a leaf node
		 */
		ExtraTrees(std::size_t nTrees = 100, std::size_t minLeafSize = 1,
			std::size_t nClasses = 2);

		/**
		 * @brief Trains the Extra-Trees model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;
	
	private:
		mlpack::RandomForest<> model_;
		std::size_t nTrees_;
		std::size_t nClasses_;
		std::size_t minLeafSize_;
	};

	/**
	 * @brief Random Forest classifier implementation using Shark
	 * Implements standard Random Forest (not Extra-Trees)
	 */
	struct RandomForest : BaseEstimator
	{
		/**
		 * @brief Constructs Random Forest classifier
		 * @param nTrees Number of trees in the forest
		 * @param minLeafSize Minimum samples required in a leaf node
		 */
		RandomForest(std::size_t nTrees = 100, std::size_t minLeafSize = 1, 
			std::size_t nClasses = 2);

		/**
		 * @brief Trains the Random Forest model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;
	
	private:
		mlpack::RandomForest<> model_;
        std::size_t nTrees_;
		std::size_t nClasses_;
        std::size_t minLeafSize_;
	};

	/**
	 * @brief K-Nearest Neighbors classifier implementation using Shark
	 * Implements standard KNN algorithm
	 */
	struct KNN : BaseEstimator
	{
		std::vector<std::vector<float>> train_features_;  // Store training data features
		std::vector<int> train_labels_;                   // Store training data labels
		std::size_t k_;                                   // Number of neighbors to consider
		std::string metric_;                              // Distance metric to use (e.g., "euclidean", "manhattan")

		/**
		 * @brief Constructs KNN classifier
		 * @param k Number of neighbors to consider
		 * @param metric Distance metric to use (default: "euclidean")
		 */
		KNN(std::size_t k = 5, std::string metric = "euclidean");

		/**
		 * @brief Trains the KNN model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;
	};

	/**
	 * @brief Logistic Regression classifier implementation using Shark
	 * Implements standard Logistic Regression
	 */
	struct LR : BaseEstimator
	{

		/**
		 * @brief Constructs Logistic Regression classifier
		 * @param lambda Regularization parameter
		 */
		LR(double lambda, std::size_t nClasses);

		/**
		 * @brief Trains the Logistic Regression model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;
	
	private:
		double lambda_;
		std::size_t nClasses_;
		mlpack::SoftmaxRegression<> model_;
	};

	/**
	 * @brief Neural Network classifier implementation using mlpack
	 * Implements a feed-forward neural network with two hidden layers
	 */
	struct NeuralNet : BaseEstimator
	{
		/**
		 * @brief Constructs Neural Network classifier
		 * @param hiddenUnits1 Number of units in the first hidden layer
		 * @param hiddenUnits2 Number of units in the second hidden layer
		 * @param nClasses Number of output classes
		 */
		NeuralNet(std::size_t hiddenUnits1 = 64, std::size_t hiddenUnits2 = 32, 
				std::size_t nClasses = 2);

		/**
		 * @brief Trains the Neural Network model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;

	private:
		mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::HeInitialization> model_;
		std::size_t hiddenUnits1_;
		std::size_t hiddenUnits2_;
		std::size_t nClasses_;
		std::size_t inputDim_;
	};

	/**
	 * @brief Support Vector Machine classifier using MLpack
	 * Implements multiclass classification with linear kernel
	 */
	struct SVM_ML : BaseEstimator
	{
		/**
		 * @brief Constructs SVM with specified parameters
		 * @param C Regularization parameter
		 * @param gamma Kernel parameter for RBF
		 */
		SVM_ML(double C = 1.0, double gamma = 0.01);

		/**
		 * @brief Trains the SVM model
		 * @param X Training data (n_samples x n_features)
		 * @param y Target labels (n_samples)
		 */
		void train(const MatrixXd &X, const VectorXi &y) override;

		/**
		 * @brief Predicts labels for test data
		 * @param X Test data (n_samples x n_features)
		 * @param y_pred Output predicted labels (n_samples)
		 */
		void predict(const MatrixXd &X, VectorXi &y_pred) override;

		/**
		 * @brief Saves the model to a file
		 * @param directory Path where to save the model
		 * @return true if successful, false otherwise
		 */
		bool save(const std::string &directory) const override;

		/**
		 * @brief Loads the model from a file
		 * @param directory Path from where to load the model
		 * @return true if successful, false otherwise
		 */
		bool load(const std::string &directory) override;

	private:
		double C_;
		double gamma_;
		size_t nClasses_;
		mlpack::LinearSVM<> model_;
	};

}