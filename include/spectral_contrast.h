#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

std::vector<float> extractSpectralContrastFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    int numberBands,
    float lowFrequencyBound,
    float highFrequencyBound,
    float neighbourRatio,
    float staticDistribution,
    essentia::standard::AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    std::vector<essentia::Real> inputAudio,
    bool appendToFeatureVector
);