#include "chroma.h"
#include "feature_utils.h"

using namespace essentia;
using namespace standard;

std::vector<Real> extractChromaFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    float minFrequency,
    int binsPerOctave,
    float threshold,
    const std::string& normalizeType,
    const std::string& windowType,
    AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    bool appendToFeatureVector
) {
    std::vector<Real> audioBuffer, frame, windowedFrame;
    Algorithm* loader = createAudioLoader(filename, sampleRate, audioBuffer);
    Algorithm* frameCutter = createFrameCutter(frameSize, hopSize, audioBuffer, frame);
    Algorithm* windowing = createWindowing(frame, windowedFrame);

    Algorithm* chroma = factory.create("Chromagram",
        "sampleRate", sampleRate,
        "minFrequency", minFrequency,
        "binsPerOctave", binsPerOctave,
        "threshold", threshold,
        "normalizeType", normalizeType,
        "windowType", windowType);

    std::vector<Real> chromaCoeffs;
    chroma->input("frame").set(windowedFrame);
    chroma->output("chromagram").set(chromaCoeffs);

    std::vector<std::vector<Real>> allChroma;
    while (true) {
        frameCutter->compute();
        if (frame.empty()) break;

        windowing->compute();
        chroma->compute();
        allChroma.push_back(chromaCoeffs);
    }

    std::vector<Real> meanChroma, stdChroma, finalVec;
    if (!allChroma.empty()) {
        computeStats(allChroma, meanChroma, stdChroma);
        finalVec.insert(finalVec.end(), meanChroma.begin(), meanChroma.end());
        finalVec.insert(finalVec.end(), stdChroma.begin(), stdChroma.end());
    }

    delete loader;
    delete frameCutter;
    delete windowing;
    delete chroma;

    if (appendToFeatureVector) {
        featureVector.insert(featureVector.end(), finalVec.begin(), finalVec.end());
    }
    return finalVec;
}