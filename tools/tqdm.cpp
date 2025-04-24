#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

class Tqdm
{
private:
    int total;
    int current;
    int barWidth;
    std::string prefix;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    bool showEta;

    void updateDisplay()
    {
        float progress = static_cast<float>(current) / static_cast<float>(total);
        int pos = static_cast<int>(barWidth * progress);

        std::cout << "\r" << prefix << " [";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }

        // Calculate ETA
        std::cout << "] " << int(progress * 100.0) << "% ";

        if (showEta && current > 0)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

            if (elapsed > 0)
            {
                double itemsPerSecond = static_cast<double>(current) / elapsed;
                int eta = static_cast<int>((total - current) / itemsPerSecond);

                // Format ETA nicely
                int etaMin = eta / 60;
                int etaSec = eta % 60;
                std::cout << "| ETA: " << etaMin << "m " << etaSec << "s";
            }
        }

        std::cout << " (" << current << "/" << total << ")";
        std::cout.flush();
    }

public:
    Tqdm(int total, const std::string &prefix = "Progress", int barWidth = 50, bool showEta = true)
        : total(total), current(0), barWidth(barWidth), prefix(prefix), showEta(showEta)
    {
        startTime = std::chrono::high_resolution_clock::now();
        updateDisplay();
    }

    void update(int steps = 1)
    {
        current += steps;
        if (current > total)
            current = total;
        updateDisplay();
    }

    void finish()
    {
        current = total;
        updateDisplay();
        std::cout << std::endl;

        // Show final statistics
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        std::cout << "Completed in " << elapsed << " seconds." << std::endl;
    }
};