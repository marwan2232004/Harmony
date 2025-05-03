#pragma once
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXi;

/**
 * @brief Base interface for all estimator classes
 * Provides virtual methods for training and prediction
 */
struct BaseEstimator
{
	virtual ~BaseEstimator() = default;

	/**
     * @brief Trains the model on given data
     * @param X Training data (n_samples x n_features)
     * @param y Target labels (n_samples)
     */
	virtual void train(const MatrixXd &X, const VectorXi &y) = 0;

	/**
     * @brief Predicts labels for given data
     * @param X Test data (n_samples x n_features)
     * @param y_pred Output predicted labels (n_samples)
     */
	virtual void predict(const MatrixXd &X, VectorXi &y_pred) = 0;

	/**
     * @brief Saves the model to a file
     * @param filepath Path where to save the model
     * @return true if successful, false otherwise
     */
    virtual bool save(const std::string &filepath) const = 0;
    
    /**
     * @brief Loads the model from a file
     * @param filepath Path from where to load the model
     * @return true if successful, false otherwise
     */
    virtual bool load(const std::string &filepath) = 0;
	
	
};

/**
 * @brief Stacking ensemble classifier
 * Combines multiple base models' predictions using a meta-model
 */
class StackingClassifier
{
	std::vector<std::unique_ptr<BaseEstimator>> bases_;
	std::unique_ptr<BaseEstimator> meta_;
	int K_, L_;
	bool fitted_ = false;
	std::mt19937 rng_;

public:
	/**
     * @brief Constructs a stacking classifier
     * @param bases Vector of base estimators
     * @param meta Meta-model that combines base predictions
     * @param n_folds Number of folds for cross-validation
     * @param seed Random seed for reproducibility
     */
	StackingClassifier(std::vector<std::unique_ptr<BaseEstimator>> bases,
					   std::unique_ptr<BaseEstimator> meta,
					   int n_folds = 5,
					   unsigned seed = 1234);

	/**
   	 * @brief Trains the stacking classifier
   	 * Uses out-of-fold predictions from base models to train meta-model
   	 * @param X Training data (n_samples x n_features)
   	 * @param y Target labels (n_samples)
   	 */
	void fit(const MatrixXd &X, const VectorXi &y);

	/**
     * @brief Predicts labels for new data
     * @param X Test data (n_samples x n_features)
     * @param out Output predicted labels (n_samples)
     * @throws std::runtime_error if model hasn't been trained
     */
	void predict(const MatrixXd &X, VectorXi &out) const;

	/**
	 * @brief Saves all models (base models and meta model) to separate files
	 * @param directory Directory where to save the models
	 * @return true if all models were saved successfully, false otherwise
	 */
	bool saveModels(const std::string &directory) const;

	/**
	 * @brief Loads all models from files
	 * @param directory Directory where models were saved
	 * @return true if all models were loaded successfully, false otherwise
	 */
	bool loadModels(const std::string &directory);
};
