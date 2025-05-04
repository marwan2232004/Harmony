#include "feature_extractor.h"

std::vector<float> getFeatureVector(std::string path, std::vector<Real> inputAudio) {
    int sampleRate = 16000;

    AlgorithmFactory& factory = AlgorithmFactory::instance();

    std::vector<float> featureVector;
    
    std::vector<float> MFCCfeatures = extractMFCCFeatures(
        path, sampleRate, 400, 160, 26, 26, 0, 8000, 22, 2, "dbamp", factory, featureVector, inputAudio,true
    );

    // std::vector<float> ChromaFeatures = extractChromaFeatures(
    //     path, sampleRate, 32768, 16384, 27.5f, 36, 0.0f, "unit_max", "hann", factory, featureVector, inputAudio, true
    // );

    // std::vector<float> SpectralContrastFeatures = extractSpectralContrastFeatures(
    //     path, sampleRate, 2048, 1024, 6, 20, 8000, 0.4f, 1.0f, factory, featureVector, inputAudio, true
    // );

    // std::vector<float> TonnetzFeatures = extractTonnetzFeatures(
    //     path, sampleRate, factory, featureVector, inputAudio, true
    // );

    // std::vector<float> MelSpectrogramFeatures = extractMelSpectrogramFeatures(
    //     path, sampleRate, 2048, 1024, 40, 20, 8000, "htkMel", "linear", "unit_sum", "power", factory, featureVector, inputAudio, true
    // );

    return featureVector;
}