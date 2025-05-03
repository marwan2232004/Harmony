#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

std::vector<float> extractMelSpectrogramFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    int numberBands,
    float lowFrequencyBound,
    float highFrequencyBound,
    const std::string& warpingFormula,
    const std::string& weighting,
    const std::string& normalize,
    const std::string& type,
    essentia::standard::AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    std::vector<essentia::Real> inputAudio,
    bool appendToFeatureVector
);