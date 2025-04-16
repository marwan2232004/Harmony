#pragma once
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <bits/stdc++.h>

using namespace essentia;
using namespace standard;

void initializeEssentia();
void shutdownEssentia();
Algorithm* createAudioLoader(const std::string& filename, int sampleRate, std::vector<Real>& audioBuffer);
Algorithm* createFrameCutter(int frameSize, int hopSize, const std::vector<Real>& audioBuffer, std::vector<Real>& frame);
Algorithm* createWindowing(const std::vector<Real>& frame, std::vector<Real>& windowedFrame);
void computeStats(const std::vector<std::vector<Real>>& features, std::vector<Real>& means, std::vector<Real>& stddevs);
