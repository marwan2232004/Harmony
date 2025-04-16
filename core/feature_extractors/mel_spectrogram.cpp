#include "mel_spectrogram.h"
#include "feature.h"

using namespace essentia;
using namespace standard;

std::vector<Real> extractMelSpectrogramFeatures(
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
) {
    std::vector<Real> audioBuffer, frame, windowedFrame;
    Algorithm* loader = createAudioLoader(filename, sampleRate, audioBuffer);
    Algorithm* frameCutter = createFrameCutter(frameSize, hopSize, audioBuffer, frame);
    Algorithm* windowing = createWindowing(frame, windowedFrame);

    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* spectrum = factory.create("Spectrum", "size", frameSize);
    std::vector<Real> spectrumFrame;
    spectrum->input("frame").set(windowedFrame);
    spectrum->output("spectrum").set(spectrumFrame);

    Algorithm* melBands = factory.create("MelBands",
        "sampleRate", sampleRate,
        "numberBands", numberBands,
        "lowFrequencyBound", lowFrequencyBound,
        "highFrequencyBound", highFrequencyBound,
        "warpingFormula", warpingFormula,
        "weighting", weighting,
        "normalize", normalize,
        "type", type);

    std::vector<Real> melBandsFrame;
    melBands->input("spectrum").set(spectrumFrame);
    melBands->output("bands").set(melBandsFrame);

    std::vector<std::vector<Real>> allMelBands;
    while (true) {
        frameCutter->compute();
        if (frame.empty()) break;

        windowing->compute();
        spectrum->compute();
        melBands->compute();
        allMelBands.push_back(melBandsFrame);
    }

    std::vector<Real> meanMelBands, stdMelBands, finalVec;
    if (!allMelBands.empty()) {
        computeStats(allMelBands, meanMelBands, stdMelBands);
        finalVec.insert(finalVec.end(), meanMelBands.begin(), meanMelBands.end());
        finalVec.insert(finalVec.end(), stdMelBands.begin(), stdMelBands.end());
    }

    delete loader;
    delete frameCutter;
    delete windowing;
    delete spectrum;
    delete melBands;

    return finalVec;
}