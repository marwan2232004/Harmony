#include "stacking_classifier.h"
#include "my_estimators.h"
#include <Eigen/Dense>

int main(){
  // load or compute your MFCC feature matrix X (n×d) and labels y (n)
  Eigen::MatrixXd X_train; 
  Eigen::VectorXi y_train;

  // instantiate base learners
  std::vector<std::unique_ptr<BaseEstimator>> base_models;
  base_models.push_back(std::make_unique<MySVM>(1000, 0.0001));
  base_models.push_back(std::make_unique<MyExtraTrees>(200, 5));
  base_models.push_back(std::make_unique<MyRandomForest>(300, 5));
  base_models.push_back(std::make_unique<MyKNN>(3));
  base_models.push_back(std::make_unique<MyLR>(0.01));

  // meta‐learner also Logistic Regression
  auto meta_model = std::make_unique<MyLR>(0.01);

  // build the stacker
  StackingClassifier stacker(
    std::move(base_models),
    std::move(meta_model),
    5,  // 5-fold
    42  // seed
  );

  // train
  stacker.fit(X_train, y_train);

  // test
  Eigen::MatrixXd X_test;
  Eigen::VectorXi y_pred;
  stacker.predict(X_test, y_pred);

  // ... evaluate y_pred vs. true labels ...
}
