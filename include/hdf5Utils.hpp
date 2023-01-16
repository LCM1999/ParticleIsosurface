#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

#include "global.h"
#include "pugixml.hpp"

bool readShonDyParticleData(const std::string &fileName,
                            std::vector<Eigen::Vector3f> &positions,
                            std::vector<float> &densities,
                            std::vector<float> &masses);

void readInt(hid_t fileID, const std::string &veclLocation,
             std::vector<int> &readVec);

void readDouble(hid_t fileID, const std::string &veclLocation,
                std::vector<double> &readVec);

int size(hid_t fileID, const std::string &groupName);