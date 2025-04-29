#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>

#include <filesystem>
namespace fs = std::filesystem;

class Logger {
public:
    // Store colors in struct for global access
    struct COLOR {
        static const std::string RESET;
        static const std::string RED;
        static const std::string GREEN;
        static const std::string YELLOW;
        static const std::string BLUE;
        static const std::string MAGENTA;
        static const std::string CYAN;
        static const std::string WHITE;
    };

    // Log levels
    enum class Level { DEBUG, INFO, WARN, ERROR };

    // Configuration structure
    struct Config {
        bool enableFileLogging = false;
        std::string logDirectory = "logs";
        std::string logFilename = "output.log";
        bool coloredOutput = true;

        Config(bool enableFileLogging = false,
            std::string logDirectory = "logs",
            std::string logFilename = "output.log",
            bool coloredOutput = true)
         : enableFileLogging(enableFileLogging),
           logDirectory(std::move(logDirectory)),
           logFilename(std::move(logFilename)),
           coloredOutput(coloredOutput) {}
    };

    /**
     * @brief Get the singleton instance of the Logger class.
     * 
     * @return Logger& Reference to the singleton instance.
     */
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    /**
     * @brief Set up UTF-8 encoding for console output.
     */
    void setupUTF8();

    /**
     * @brief Initialize the logger with application name and configuration.
     * 
     * @param appName Name of the application.
     * @param config Configuration settings for the logger.
     */
    void initialize(const std::string& appName, const Config& config);

    /**
     * @brief Log a message with a specific log level.
     * 
     * @param message The message to log.
     * @param level The log level (DEBUG, INFO, WARN, ERROR).
     * @param logToFile Whether to log to file (default: true).
     */
    void log(const std::string& message, Level level = Level::INFO, bool logToFile = true);

    /**
     * @brief Log a message with a specific log level and color.
     * 
     * @param message The message to log.
     * @param level The log level (DEBUG, INFO, WARN, ERROR).
     */
    void logArguments(const std::vector<std::string>& args);

    /**
     * @brief Log a configuration parameter.
     * 
     * @param name The name of the configuration parameter.
     * @param value The value of the configuration parameter.
     */
    template<typename T>
    void logConfig(const std::string& name, const T& value);

    // Progress bar class
    class ProgressBar {
    public:
        ProgressBar(int total, const std::string& description = "")
            : total(total), description(description), startTime(std::chrono::steady_clock::now()) {
            update(0);
        }

        void update(int current) {
            std::lock_guard<std::mutex> lock(Logger::getInstance().logMutex);
            float progress = static_cast<float>(current) / total;
            int barWidth = 50;
            int pos = barWidth * progress;

            std::stringstream ss;
            ss << "\r" << description << " [";
            for(int i = 0; i < barWidth; ++i) {
                ss << (i < pos ? "=" : (i == pos ? ">" : " "));
            }
            ss << "] " << int(progress * 100.0) << "% ";

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            
            if(current > 0) {
                double itemsPerSecond = static_cast<double>(current) / elapsed;
                int eta = static_cast<int>((total - current) / itemsPerSecond);
                ss << "| ETA: " << eta/60 << "m " << eta%60 << "s";
            }

            ss << " (" << current << "/" << total << ")";
            
            if(Logger::getInstance().config.coloredOutput) {
                std::cout << COLOR::CYAN << ss.str() << COLOR::RESET;
            } else {
                std::cout << ss.str();
            }
            std::cout.flush();
        }

        void finish() {
            update(total);
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
            std::cout << "\nCompleted in " << duration.count() << " seconds." << std::endl;
        }

    private:
        int total;
        std::string description;
        std::chrono::steady_clock::time_point startTime;
    };

private:
    std::string appName;
    Config config;
    std::ofstream logFile;
    std::mutex logMutex;

    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Format the log message with timestamp and level.
     * 
     * @param message The message to format.
     * @param level The log level.
     * @return Formatted log message.
     */
    std::string formatLog(const std::string& message, Level level) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
           << " [" << levelToString(level) << "] "
           << message;
        return ss.str();
    }

    /**
     * @brief Convert log level to string.
     * 
     * @param level The log level.
     * @return String representation of the log level.
     */
    std::string levelToString(Level level) {
        switch(level) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO: return "INFO";
            case Level::WARN: return "WARN";
            case Level::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief Get the color code for a specific log level.
     * 
     * @param level The log level.
     * @return Color code as a string.
     */
    std::string levelColor(Level level) {
        if(!config.coloredOutput) return "";
        switch(level) {
            case Level::DEBUG: return COLOR::CYAN;
            case Level::INFO: return COLOR::GREEN;
            case Level::WARN: return COLOR::YELLOW;
            case Level::ERROR: return COLOR::RED;
            default: return COLOR::RESET;
        }
    }

    /**
     * @brief Strip ANSI color codes from a string.
     * 
     * @param text The input string with color codes.
     * @return String without color codes.
     */
    std::string stripColors(const std::string& text) {
        std::string result;
        result.reserve(text.size());

        for(size_t i = 0; i < text.size(); ++i) {
            if(text[i] == '\033') {
                while(i < text.size() && text[i] != 'm') ++i;
            } else {
                result += text[i];
            }
        }
        return result;
    }
};