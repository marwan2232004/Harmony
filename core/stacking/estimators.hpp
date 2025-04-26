#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <dlib/svm_threaded.h>
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
#include <shark/Data/Csv.h>
#include <eigen3/Eigen/Dense>
#include "stacking_classifier.hpp"

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
		using ovo_trainer_type = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
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
		void predict(const MatrixXd &X, VectorXi &y_pred) const override;

	private:
		df_type decision_function_;
	};

	/**
	 * @brief Extra-Trees classifier implementation using Shark
	 * Note: Extremely Randomized Trees variant
	 */
	struct ExtraTrees : BaseEstimator
	{
		// using ETTrainer = shark::ForestTrainer<
		// 	shark::DecisionTree<shark::RealVector>,
		// 	ExtraTreeTrainer<shark::RealVector>
		// >;
		shark::RFTrainer<shark::RealVector> trainer; // Random Forest trainer configured for Extra-Trees
		shark::RFClassifier<shark::RealVector> model;	  // Trained Extra-Trees model

		/**
		 * @brief Constructs Extra-Trees classifier
		 * @param nTrees Number of trees in the forest
		 * @param minLeafSize Minimum samples required in a leaf node
		 */
		ExtraTrees(std::size_t nTrees = 100, std::size_t minLeafSize = 1);

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
		void predict(const MatrixXd &X, VectorXi &y_pred) const override;
	};

	/**
	 * @brief Random Forest classifier implementation using Shark
	 * Implements standard Random Forest (not Extra-Trees)
	 */
	struct RandomForest : BaseEstimator
	{
		shark::RFTrainer<unsigned int> trainer;
		shark::RFClassifier<unsigned int> model;

		/**
		 * @brief Constructs Random Forest classifier
		 * @param nTrees Number of trees in the forest
		 * @param minLeafSize Minimum samples required in a leaf node
		 */
		RandomForest(std::size_t nTrees = 100, std::size_t minLeafSize = 1);

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
		void predict(const MatrixXd &X, VectorXi &y_pred) const override;
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

		/**
		 * @brief Constructs KNN classifier
		 * @param k Number of neighbors to consider
		 */
		KNN(std::size_t k = 5);

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
		void predict(const MatrixXd &X, VectorXi &y_pred) const override;
	};

	/**
	 * @brief Logistic Regression classifier implementation using Shark
	 * Implements standard Logistic Regression
	 */
	struct LR : BaseEstimator
	{
		shark::LogisticRegression<shark::RealVector> trainer;
		shark::LogisticRegression<shark::RealVector>::ModelType model;
		double lambda_, lambda2_;

		/**
		 * @brief Constructs Logistic Regression classifier
		 * @param lambda1 Regularization parameter
		 * @param lambda2 Regularization parameter
		 */
		LR(double lambda1 = 1e-4, double lambda2 = 1e-4);

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
		void predict(const MatrixXd &X, VectorXi &y_pred) const override;
	};

}