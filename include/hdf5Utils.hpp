#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

#include "global.h"
#include "pugixml.hpp"

bool readShonDyParticleXDMF(const std::string dir_path,
                            const std::string xdmf_file,
                            std::vector<std::string> &files);

bool readShonDyParticleData(const std::string &fileName,
                            std::vector<Eigen::Vector3d> &positions,
                            std::vector<double>* radiuses, const double scale);

void readInt(hid_t fileID, const std::string &veclLocation,
             std::vector<int> &readVec);

void readDouble(hid_t fileID, const std::string &veclLocation,
                std::vector<double> &readVec);

int size(hid_t fileID, const std::string &groupName);