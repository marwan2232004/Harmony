#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

std::vector<float> extractChromaFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    float minFrequency,
    int binsPerOctave,
    float threshold,
    const std::string& normalizeType,
    const std::string& windowType,
    essentia::standard::AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    bool appendToFeatureVector
);