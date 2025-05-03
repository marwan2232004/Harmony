#include "spectral_contrast.h"
#include "feature_utils.h"

using namespace essentia;
using namespace standard;

std::vector<Real> extractSpectralContrastFeatures(
    const std::string& filename,
    int sampleRate,
    int frameSize,
    int hopSize,
    int numberBands,
    float lowFrequencyBound,
    float highFrequencyBound,
    float neighbourRatio,
    float staticDistribution,
    AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    std::vector<Real> inputAudio,
    bool appendToFeatureVector
) {
    std::vector<Real> audioBuffer, frame, windowedFrame;
    if(inputAudio.empty()) {
        // Load audio file
        Algorithm* loader = createAudioLoader(filename, sampleRate, audioBuffer);
        if (audioBuffer.empty()) {
            std::cerr << "Error loading audio file: " << filename << std::endl;
            return {};
        }
        delete loader;
    } else {
        audioBuffer = inputAudio;
    }
    Algorithm* frameCutter = createFrameCutter(frameSize, hopSize, audioBuffer, frame);
    Algorithm* windowing = createWindowing(frame, windowedFrame);

    Algorithm* spectrum = factory.create("Spectrum", "size", frameSize);
    std::vector<Real> spectrumFrame;
    spectrum->input("frame").set(windowedFrame);
    spectrum->output("spectrum").set(spectrumFrame);

    Algorithm* spectralContrast = factory.create("SpectralContrast",
        "sampleRate", sampleRate,
        "numberBands", numberBands,
        "lowFrequencyBound", lowFrequencyBound,
        "highFrequencyBound", highFrequencyBound,
        "neighbourRatio", neighbourRatio,
        "staticDistribution", staticDistribution);

    std::vector<Real> scValleys, scPeaks;
    spectralContrast->input("spectrum").set(spectrumFrame);
    spectralContrast->output("spectralContrast").set(scPeaks);
    spectralContrast->output("spectralValley").set(scValleys);

    std::vector<std::vector<Real>> allSCFeatures;
    while (true) {
        frameCutter->compute();
        if (frame.empty()) break;

        windowing->compute();
        spectrum->compute();
        spectralContrast->compute();
        
        std::vector<Real> frameFeatures;
        frameFeatures.insert(frameFeatures.end(), scPeaks.begin(), scPeaks.end());
        frameFeatures.insert(frameFeatures.end(), scValleys.begin(), scValleys.end());
        allSCFeatures.push_back(frameFeatures);
    }

    std::vector<Real> meanFeatures, stdFeatures, finalVec;
    if (!allSCFeatures.empty()) {
        computeStats(allSCFeatures, meanFeatures, stdFeatures);
        finalVec.insert(finalVec.end(), meanFeatures.begin(), meanFeatures.end());
        finalVec.insert(finalVec.end(), stdFeatures.begin(), stdFeatures.end());
    }

    delete frameCutter;
    delete windowing;
    delete spectrum;
    delete spectralContrast;

    if (appendToFeatureVector) {
        featureVector.insert(featureVector.end(), finalVec.begin(), finalVec.end());
    }
    return finalVec;

}