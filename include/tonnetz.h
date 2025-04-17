#pragma once
#include <essentia/essentia.h>
#include <bits/stdc++.h>

std::vector<essentia::Real> extractTonnetzFeatures(
    const std::string& filename,
    int sampleRate
);