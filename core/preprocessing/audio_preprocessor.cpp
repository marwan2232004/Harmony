#include "audio_preprocessor.hpp"


using namespace essentia;
using namespace standard;

namespace fs = std::filesystem;

AudioPreprocessor::AudioPreprocessor(float targetDuration) 
    : targetDuration(targetDuration) {
    // Initialize Essentia
    essentia::init();
}

AudioPreprocessor::~AudioPreprocessor() {
    // Shut down Essentia
    essentia::shutdown();
}

bool AudioPreprocessor::processFile(const std::string& inputPath, const std::string& outputPath, float& duration) {
    if (!fs::exists(inputPath)) {
        std::cerr << "Input file does not exist: " << inputPath << std::endl;
        return false;
    }
    
    try {
        int sampleRate = 0;
        std::vector<Real> audioBuffer = AudioUtil::readAudioFile(inputPath, duration ,sampleRate);

        if (audioBuffer.empty()) {
            return false;
        }

        // Resample to 16kHz if necessary
        const int targetSampleRate = 16000;
        if (sampleRate != targetSampleRate) {
            
            AlgorithmFactory& factory = AlgorithmFactory::instance();
            Algorithm* resampler = factory.create("Resample",
                                            "inputSampleRate", sampleRate,
                                            "outputSampleRate", targetSampleRate,
                                            "quality", 1);  // 1 is high quality
            
            std::vector<Real> resampledBuffer;
            resampler->input("signal").set(audioBuffer);
            resampler->output("signal").set(resampledBuffer);
            resampler->compute();
            
            // Replace original buffer with resampled buffer
            audioBuffer = resampledBuffer;
            sampleRate = targetSampleRate;
            
            delete resampler;
        }
        
        // Apply processing steps according to enabled flags
        if (silenceRemovalEnabled) {
            removeSilence(audioBuffer, sampleRate);
        }
        
        if (noiseReductionEnabled) {
            reduceNoise(audioBuffer);
        }
        
        if (normalizeEnabled) {
            normalizeVolume(audioBuffer);
        }
        
        if (trimEnabled) {
            trimAudio(audioBuffer, sampleRate);
            if (audioBuffer.empty()) {
                return false;
            }
        }

        duration = static_cast<float>(audioBuffer.size()) / static_cast<float>(sampleRate);
        
        // Create output directory if it doesn't exist
        fs::path outputDir = fs::path(outputPath).parent_path();
        if (!outputDir.empty() && !fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
        
        // Write the processed audio
        return writeAudioFile(audioBuffer, sampleRate, outputPath);
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing file " << inputPath << ": " << e.what() << std::endl;
        return false;
    }
}


std::vector<std::string> getTokens(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

void saveMetadata(const std::string& outputPath, const std::vector<std::string>& metadata) {
    std::string cleanedMetadataFilePath = outputPath + "/processed_metadata.tsv";
    std::ofstream cleanedFile(cleanedMetadataFilePath);
    if (!cleanedFile.is_open())
    {
        throw std::runtime_error("Could not create cleaned metadata file: " + cleanedMetadataFilePath);
    }

    // Write header
    cleanedFile << "path\tage\tgender\tduration\n";
    Tqdm saveProgress(metadata.size(), "Saving cleaned metadata");
    for (const auto &metadata : metadata)
    {
        cleanedFile << metadata << "\n";
        saveProgress.update();
    }

    cleanedFile.close();
    saveProgress.finish();
}

// input format:  client_id	path	sentence	up_votes	down_votes	age	gender	accent	label
std::vector<std::string> AudioPreprocessor::processBatch(
    const std::string& metadataPath,
    const std::string& outputDir,
    const int maxFiles,
    bool showProgress) {
    
    std::vector<std::string> outputPaths;
    std::vector<std::string> outputMatadata;

    std::string dataPath = metadataPath.substr(0, metadataPath.find_last_of("/"));

    std::ifstream file(metadataPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open metadata file: " + metadataPath);
    }

    // Count number of lines for progress bar
    int lineCount = 0;
    std::string line;
    while (std::getline(file, line))
    {
        lineCount++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);
    
    lineCount = std::min(lineCount, maxFiles);

    // Use Tqdm for progress tracking
    Tqdm tqdm(lineCount, "Processing audio files");
    
    while(std::getline(file,line)) {
        if (lineCount <= 0) {
            break;
        }
   
        std::vector<std::string> tokens = getTokens(line, '\t');
        if (tokens.size() < 7) {
            std::cerr << "Invalid line format: " << line << std::endl;
            continue;
        }

        fs::path inputFile = fs::path(tokens[1]);
        fs::path outputPath = fs::path(outputDir) / inputFile.filename();
        outputPath.replace_extension(".wav"); // For consistency
        
        float duration = -1.0f;
        bool success = processFile(dataPath + "/" + tokens[1], outputPath.string(), duration);

        if (success) {
            std::string cleanedLine = tokens[1] + "\t" + tokens[5] + "\t" + tokens[6] + "\t" + std::to_string(duration);
            outputMatadata.push_back(cleanedLine);
            outputPaths.push_back(outputPath.string());
        }
        
        // Update progress using Tqdm
        if (showProgress) {
            tqdm.update();
        }

        lineCount--;
    }

    file.close();
    
    // Finish the progress bar
    if (showProgress) {
        tqdm.finish();
    }

    saveMetadata(outputDir, outputMatadata);

    std::cout << "Kept " << outputMatadata.size() << " valid files out of " << maxFiles << " total entries" << std::endl;
    return outputPaths;
}

// Individual processing functions

void AudioPreprocessor::trimAudio(std::vector<essentia::Real>& audioBuffer, int sampleRate) {
    // Calculate number of samples to keep
    int targetSamples = static_cast<int>(targetDuration * sampleRate);
    
    // If the buffer is longer than the target, trim it
    if (int(audioBuffer.size()) > targetSamples) {
        audioBuffer.resize(targetSamples);
    } 
    else if (int(audioBuffer.size()) < targetSamples) {
        audioBuffer.clear();
    }
}

void AudioPreprocessor::normalizeVolume(std::vector<essentia::Real>& audioBuffer) {
    if (audioBuffer.empty()) {
        return;
    }
    
    // Calculate current RMS
    float currentRMS = calculateRMS(audioBuffer);
    
    // If RMS is close to zero, avoid division by zero
    if (currentRMS < 1e-6) {
        return;
    }
    
    // Calculate scaling factor
    float scaleFactor = targetRMS / currentRMS;
    
    // Apply gain
    for (auto& sample : audioBuffer) {
        sample *= scaleFactor;
        
        // Apply soft clipping to avoid distortion if needed
        if (sample > 0.95f) {
            sample = 0.95f;
        } else if (sample < -0.95f) {
            sample = -0.95f;
        }
    }
}

void AudioPreprocessor::reduceNoise(std::vector<essentia::Real>& audioBuffer) {
    if (audioBuffer.empty()) {
        return;
    }
    
    try {
        // Create spectral gate algorithm using Essentia
        AlgorithmFactory& factory = AlgorithmFactory::instance();
        
        // Convert to spectrum
        Algorithm* fftwfft = factory.create("FFT");
        Algorithm* window = factory.create("Windowing", "type", "hann");
        Algorithm* ifft = factory.create("IFFT");
        
        const int frameSize = 2048;
        const int hopSize = 1024;
        
        // Process frame by frame
        for (int i = 0; i < int(audioBuffer.size()); i += hopSize) {
            // Extract frame
            std::vector<Real> frame;
            for (int j = 0; j < frameSize; ++j) {
                if (i + j < int(audioBuffer.size())) {
                    frame.push_back(audioBuffer[i + j]);
                } else {
                    frame.push_back(0.0f); // Zero padding
                }
            }
            
            // Apply window
            std::vector<Real> windowedFrame;
            window->input("frame").set(frame);
            window->output("frame").set(windowedFrame);
            window->compute();
            
            // Calculate FFT
            std::vector<std::complex<Real>> spectrum;
            fftwfft->input("frame").set(windowedFrame);
            fftwfft->output("fft").set(spectrum);
            fftwfft->compute();
            
            // Apply spectral gating (simple noise reduction)
            for (auto& bin : spectrum) {
                Real magnitude = std::abs(bin);
                if (magnitude < noiseThreshold) {
                    bin = std::complex<Real>(0.0f, 0.0f);
                }
            }
            
            // Inverse FFT
            std::vector<Real> processedFrame;
            ifft->input("fft").set(spectrum);
            ifft->output("frame").set(processedFrame);
            ifft->compute();
            
            // Overlap-add to output buffer
            for (int j = 0; j < frameSize; ++j) {
                if (i + j < int(audioBuffer.size())) {
                    audioBuffer[i + j] = processedFrame[j];
                }
            }
        }
        
        // Clean up algorithms
        delete fftwfft;
        delete window;
        delete ifft;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in noise reduction: " << e.what() << std::endl;
    }
}

void AudioPreprocessor::removeSilence(std::vector<essentia::Real>& audioBuffer, int sampleRate) {
    if (audioBuffer.empty()) {
        return;
    }
    
    // Calculate number of samples for minimum silence duration
    int minSilenceSamples = static_cast<int>(minSilenceMs * sampleRate / 1000);
    
    // Find silent segments
    std::vector<std::pair<int, int>> silentSegments;
    int silenceStart = 0;
    bool inSilence = false;
    
    for (int i = 0; i < int(audioBuffer.size()); ++i) {
        float amplitude = std::abs(audioBuffer[i]);
        
        if (amplitude < silenceThreshold) {
            // Entering silence
            if (!inSilence) {
                silenceStart = i;
                inSilence = true;
            }
        } else {
            // Exiting silence
            if (inSilence) {
                int silenceEnd = i;
                int silenceLength = silenceEnd - silenceStart;
                
                // If silence is long enough, mark it for removal
                if (silenceLength >= minSilenceSamples) {
                    silentSegments.push_back(std::make_pair(silenceStart, silenceEnd));
                }
                
                inSilence = false;
            }
        }
    }
    
    // If we end with silence, add that segment too
    if (inSilence) {
        int silenceLength = int(audioBuffer.size()) - silenceStart;
        if (silenceLength >= minSilenceSamples) {
            silentSegments.push_back(std::make_pair(silenceStart, int(audioBuffer.size())));
        }
    }
    
    // If no silent segments found, return
    if (silentSegments.empty()) {
        return;
    }
    
    // Create new buffer without the silent segments
    std::vector<Real> processedBuffer;
    int lastEnd = 0;
    
    for (const auto& segment : silentSegments) {
        // Add audio before this silent segment
        processedBuffer.insert(
            processedBuffer.end(),
            audioBuffer.begin() + lastEnd,
            audioBuffer.begin() + segment.first
        );
        
        lastEnd = segment.second;
    }
    
    // Add the final portion after the last silent segment
    if (lastEnd < int(audioBuffer.size())) {
        processedBuffer.insert(
            processedBuffer.end(),
            audioBuffer.begin() + lastEnd,
            audioBuffer.end()
        );
    }
    
    // Replace original buffer with processed buffer
    audioBuffer = processedBuffer;
}

// Utility methods

float AudioPreprocessor::calculateRMS(const std::vector<essentia::Real>& buffer) {
    float sumSquared = 0.0f;
    for (const auto& sample : buffer) {
        sumSquared += sample * sample;
    }
    return std::sqrt(sumSquared / buffer.size());
}

bool AudioPreprocessor::writeAudioFile(const std::vector<essentia::Real>& buffer, int sampleRate, const std::string& filePath) {
    try {
        // Get algorithm factory
        AlgorithmFactory& factory = AlgorithmFactory::instance();
        
        // Create output directory if it doesn't exist
        fs::path outputDir = fs::path(filePath).parent_path();
        if (!outputDir.empty() && !fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
        
        // Create audio writer
        Algorithm* audioWriter = factory.create("MonoWriter",
                                             "filename", filePath,
                                             "sampleRate", sampleRate,
                                             "format", "wav",
                                            "bitrate", 32);
        
        // Set input
        audioWriter->input("audio").set(buffer);
        
        // Compute
        audioWriter->compute();
        
        // Clean up
        delete audioWriter;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing audio file: " << e.what() << std::endl;
        return false;
    }
}