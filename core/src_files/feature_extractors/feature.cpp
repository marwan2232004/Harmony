#include "feature.h"

using namespace essentia;
using namespace standard;

void initializeEssentia() {
    essentia::init();
}

void shutdownEssentia() {
    essentia::shutdown();
}

Algorithm* createAudioLoader(const std::string& filename, int sampleRate, std::vector<Real>& audioBuffer) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* loader = factory.create("MonoLoader", "filename", filename, "sampleRate", sampleRate);
    loader->output("audio").set(audioBuffer);
    loader->compute();
    return loader;
}

Algorithm* createFrameCutter(int frameSize, int hopSize, const std::vector<Real>& audioBuffer, std::vector<Real>& frame) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* frameCutter = factory.create("FrameCutter",
        "frameSize", frameSize,
        "hopSize", hopSize,
        "startFromZero", true);
    frameCutter->input("signal").set(audioBuffer);
    frameCutter->output("frame").set(frame);
    return frameCutter;
}

Algorithm* createWindowing(const std::vector<Real>& frame, std::vector<Real>& windowedFrame) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* windowing = factory.create("Windowing", "type", "hann", "normalized", false);
    windowing->input("frame").set(frame);
    windowing->output("frame").set(windowedFrame);
    return windowing;
}

void computeStats(const std::vector<std::vector<Real>>& features,
                std::vector<Real>& means,
                std::vector<Real>& stddevs) {
    int numCoeffs = features[0].size();
    means.assign(numCoeffs, 0.0);
    stddevs.assign(numCoeffs, 0.0);

    for (const auto& frame : features)
        for (int i = 0; i < numCoeffs; ++i)
            means[i] += frame[i];

    for (auto& m : means) m /= features.size();

    for (const auto& frame : features)
        for (int i = 0; i < numCoeffs; ++i)
            stddevs[i] += (frame[i] - means[i]) * (frame[i] - means[i]);

    for (auto& s : stddevs) s = sqrt(s / features.size());
}