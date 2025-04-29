#include "logger.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <locale>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

const std::string Logger::COLOR::RESET = "\033[0m";
const std::string Logger::COLOR::RED = "\033[31m";
const std::string Logger::COLOR::GREEN = "\033[32m";
const std::string Logger::COLOR::YELLOW = "\033[33m";
const std::string Logger::COLOR::BLUE = "\033[34m";
const std::string Logger::COLOR::MAGENTA = "\033[35m";
const std::string Logger::COLOR::CYAN = "\033[36m";
const std::string Logger::COLOR::WHITE = "\033[37m";


void Logger::setupUTF8() {
    try {
#ifdef _WIN32
        // Windows-specific setup
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        // Set global locale to UTF-8
        std::locale utf8_locale(".UTF-8"); // Windows style
        std::locale::global(utf8_locale);
        std::wcout.imbue(utf8_locale);
#else
        // Linux / macOS setup
        std::locale utf8_locale("C.UTF-8"); // More portable than en_US.UTF-8
        std::locale::global(utf8_locale);
        std::wcout.imbue(utf8_locale);
#endif
    } catch (const std::exception& e) {
        std::cerr << "Failed to set UTF-8 locale: " << e.what() << std::endl;
    }
}

void Logger::initialize(const std::string& appName, const Config& config) {
    this->appName = appName;
    this->config = config;
    
    if(config.enableFileLogging) {
        fs::create_directories(config.logDirectory);
        logFile.open(fs::path(config.logDirectory) / config.logFilename, 
                   std::ios::out | std::ios::app);
        log("Logger initialized. Logging to: " + 
            (fs::path(config.logDirectory) / config.logFilename).string(), Level::INFO);
    }
}

void Logger::log(const std::string& message, Level level = Level::INFO, bool logToFile = true) {
    std::lock_guard<std::mutex> lock(logMutex);
    std::string formatted = formatLog(message, level);
    
    if(config.coloredOutput) {
        std::cout << levelColor(level) << formatted << COLOR::RESET << std::endl;
    } else {
        std::cout << formatted << std::endl;
    }

    if(config.enableFileLogging && logFile.is_open()) {
        logFile << stripColors(formatted) << std::endl;
    }
}

void Logger::logArguments(const std::vector<std::string>& args) {
    std::stringstream ss;
    ss << "Command-line arguments (" << args.size() << "):\n";
    for(const auto& arg : args) {
        ss << "▸ " << arg << "\n";
    }
    log(ss.str(), Level::INFO, /*logToFile=*/true);
}

template<typename T>
void Logger::logConfig(const std::string& name, const T& value) {
    std::stringstream ss;
    ss << "Configuration: " << name << " = " << value;
    log(ss.str(), Level::INFO);
}