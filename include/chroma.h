#pragma once
#include <essentia/essentia.h>
#include <bits/stdc++.h>

std::vector<essentia::Real> extractChromaFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    float minFrequency,
    int binsPerOctave,
    float threshold,
    const std::string& normalizeType,
    const std::string& windowType
);