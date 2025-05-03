#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

std::vector<float> extractTonnetzFeatures(
    const std::string& filename,
    int sampleRate,
    essentia::standard::AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    std::vector<essentia::Real> inputAudio,
    bool appendToFeatureVector
);