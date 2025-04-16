#pragma once
#include <essentia/essentia.h>
#include <bits/stdc++.h>

std::vector<essentia::Real> extractSpectralContrastFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    int numberBands,
    float lowFrequencyBound,
    float highFrequencyBound,
    float neighbourRatio,
    float staticDistribution  
);