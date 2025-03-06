#include <utils_helper.h>

namespace cstoneOctree{

void generateGaussianParticles(const std::string& filePath, int particleCount, 
                               double mean, double stddev) {
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::normal_distribution<> dist(mean, stddev);
    
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "error to open file: " << filePath << std::endl;
        return;
    }

    for(int i = 0; i < particleCount; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        outFile << x << " " << y << " " << z << "\n";
    }

    outFile.close();
}

void readCoordinatesFromFile(const std::string& filename, std::vector<float>& r, std::vector<float>& x, std::vector<float>& y, std::vector<float>& z) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    std::string line;
    std::getline(inFile, line);
    float ri, xi, yi, zi;
    char comma;
    while(std::getline(inFile, line)){
        std::istringstream iss(line);
        iss >> ri >> comma >> xi >> comma >> yi >> comma >> zi;
        r.push_back(ri);
        x.push_back(xi);
        y.push_back(yi);
        z.push_back(zi);
    }
    inFile.close();
    std::cout << "Coordinates read from " << filename << std::endl;
}

void readMortonCodesFromFile(const std::string& filename, std::vector<uint64_t>& mortonCodes) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    uint64_t code;
    while (inFile >> code) {
        mortonCodes.push_back(code);
    }

    inFile.close();
    std::cout << "Morton codes read from " << filename << std::endl; 
}

}