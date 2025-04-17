#pragma once
#include <essentia/essentia.h>
#include <bits/stdc++.h>

std::vector<essentia::Real> extractMFCCFeatures(
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
    const std::string& logType
);
