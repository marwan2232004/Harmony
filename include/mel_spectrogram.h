#pragma once
#include <essentia/essentia.h>
#include <bits/stdc++.h>

std::vector<essentia::Real> extractMelSpectrogramFeatures(
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
    const std::string& type
);