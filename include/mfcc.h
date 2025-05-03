#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

std::vector<float> extractMFCCFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    int numberBands,
    int numberCoefficients,
    float lowFrequencyBound,
    float highFrequencyBound,
    int liftering,
    int dctType,
    const std::string& logType,
    essentia::standard::AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    std::vector<essentia::Real> inputAudio,
    bool appendToFeatureVector
);
