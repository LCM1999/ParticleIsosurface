#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <coord_struct.h>

namespace cstoneOctree{

using TreeNodeIndex = int;
void generateGaussianParticles(const std::string& filePath, int particleCount, 
                               double mean = 1.0, double stddev = 1.0);


void readCoordinatesFromFile(const std::string& filename, std::vector<float>& r, std::vector<float>& x, std::vector<float>& y, std::vector<float>& z);

void readMortonCodesFromFile(const std::string& filename, std::vector<uint64_t>& mortonCodes);

template <class T>
void writeCoordinatesToFile(const std::string& filePath, const std::vector<Vec3<T>>& coords) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "error to open file: " << filePath << std::endl;
        return;
    }

    for(int i = 0; i < coords.size(); ++i) {
        T x = coords[i].x;
        T y = coords[i].y;
        T z = coords[i].z;
        outFile << x << " " << y << " " << z << "\n";
    }
    outFile.close();
}

template <class T>
void writeMortonCodesToFile(const std::string& filePath, const std::vector<T>& mortonCodes) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "error to open file: " << filePath << std::endl;
        return;
    }

    for(int i = 0; i < mortonCodes.size(); ++i) {
        outFile << mortonCodes[i] << "\n";
    }
    outFile.close();
}


}