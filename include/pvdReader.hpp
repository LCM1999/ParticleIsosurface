#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "pugixml.hpp"

bool readShonDyParticlesPVD(
    const std::string dir_path,
    const std::string pvd_file,
    bool &is_const_radius,
    float &radius,
    std::vector<std::string> &files
);

bool readVTU(
    const std::string &vtuFile,
    std::vector<Eigen::Vector3f> &positions,
    std::vector<float> *radiuses
);
