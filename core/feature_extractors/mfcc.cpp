#include "mfcc.h"
#include <essentia/algorithmfactory.h>

using namespace essentia;
using namespace standard;

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
    const std::string& logType
) {
    essentia::init();
    AlgorithmFactory& factory = AlgorithmFactory::instance();

    Algorithm* loader = factory.create("MonoLoader", "filename", filename, "sampleRate", sampleRate);
    std::vector<Real> audioBuffer;
    loader->output("audio").set(audioBuffer);
    loader->compute();

    Algorithm* frameCutter = factory.create("FrameCutter",
        "frameSize", frameSize,
        "hopSize", hopSize,
        "startFromZero", true);
    std::vector<Real> frame;
    frameCutter->input("signal").set(audioBuffer);
    frameCutter->output("frame").set(frame);

    Algorithm* windowing = factory.create("Windowing", "type", "hamming", "normalized", false);
    std::vector<Real> windowedFrame;
    windowing->input("frame").set(frame);
    windowing->output("frame").set(windowedFrame);

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

    std::vector<Real> mfccCoeffs, mfccBands;
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
    essentia::shutdown();

    return finalVec;
}
