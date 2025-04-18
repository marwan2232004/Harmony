#include "feature_extractor.h"
#include "feature_utils.h"
#include "svm.h"
#include "knn.h"
#include "lda.h"

using namespace essentia;
using namespace standard;

int main() {
    initializeEssentia();

    std::vector<std::vector<float>> training_features;
    std::vector<int> training_labels;

    // Should be done for all audio files
    training_features.push_back(getFeatureVector("audio/input.wav"));
    training_labels.push_back(1); // currently 0 and 1 for male and female

    train_svm_rbf("models/svm_model.dat", training_features, training_labels, 10.0f, 0.1f);
    train_lda("models/lda_model.dat", training_features, training_labels, 1.0f);
    //TODO: Implement MLP training

    // example for SVM prediction
    /*
    decision_func loaded_svm;
    dlib::deserialize("models/svm_model.dat") >> loaded_svm;
    int predicted_svm = predict_svm_rbf(loaded_svm, featureVector);
    std::cout << "SVM predicted class: " << predicted_svm << std::endl;
    */

    // example for MLP prediction
    /*
    TODO: Implement MLP prediction
    */
    

    // example for LDA prediction
    /*
    lda_model loaded_lda;
    dlib::deserialize("models/lda_model.dat") >> loaded_lda;
    int predicted_lda = predict_lda(loaded_lda, featureVector);
    std::cout << "LDA predicted class: " << predicted_lda << std::endl;
    */

    // example for KNN prediction
    /*
    int predicted_knn = predict_knn(training_features, training_labels, featureVector, 3);
    std::cout << "KNN predicted class: " << predicted_knn << std::endl;
    */

    shutdownEssentia();
    return 0;
}
