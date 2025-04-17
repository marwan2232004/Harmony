#include "mfcc.h"
#include "feature_utils.h"

using namespace essentia;
using namespace standard;

std::vector<Real> extractMFCCFeatures(
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
    const std::string& logType,
    AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    bool appendToFeatureVector
) {
    std::vector<Real> audioBuffer, frame, windowedFrame;
    Algorithm* loader = createAudioLoader(filename, sampleRate, audioBuffer);
    Algorithm* frameCutter = createFrameCutter(frameSize, hopSize, audioBuffer, frame);
    Algorithm* windowing = createWindowing(frame, windowedFrame);

    Algorithm* spectrum = factory.create("Spectrum", "size", frameSize);
    std::vector<Real> spectrumFrame;
    spectrum->input("frame").set(windowedFrame);
    spectrum->output("spectrum").set(spectrumFrame);

    Algorithm* mfcc = factory.create("MFCC",
        "inputSize", frameSize / 2 + 1,
        "sampleRate", sampleRate,
        "numberBands", numberBands,
        "numberCoefficients", numberCoefficients,
        "lowFrequencyBound", lowFrequencyBound,
        "highFrequencyBound", highFrequencyBound,
        "dctType", dctType,
        "liftering", liftering,
        "logType", logType);

    std::vector<Real> mfccCoeffs;
    std::vector<Real> mfccBands;
    mfcc->input("spectrum").set(spectrumFrame);
    mfcc->output("mfcc").set(mfccCoeffs);
    mfcc->output("bands").set(mfccBands);

    std::vector<std::vector<Real>> allMFCCs;
    while (true) {
        frameCutter->compute();
        if (frame.empty()) break;

        windowing->compute();
        spectrum->compute();
        mfcc->compute();
        allMFCCs.push_back(mfccCoeffs);
    }

    std::vector<Real> meanMFCCs, stdMFCCs, finalVec;
    if (!allMFCCs.empty()) {
        computeStats(allMFCCs, meanMFCCs, stdMFCCs);
        finalVec.insert(finalVec.end(), meanMFCCs.begin(), meanMFCCs.end());
        finalVec.insert(finalVec.end(), stdMFCCs.begin(), stdMFCCs.end());
    }

    delete loader;
    delete frameCutter;
    delete windowing;
    delete spectrum;
    delete mfcc;

    if (appendToFeatureVector) {
        featureVector.insert(featureVector.end(), finalVec.begin(), finalVec.end());
    }
    return finalVec;
}