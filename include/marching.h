#pragma once
#ifndef MARCHING_H
#define MARCHING_H

#include <vector>
#include <string.h>
#include <memory>

#include "timer.h"
#include "utils.h"
#include "iso_common.h"


class HashGrid;
class MultiLevelSearcher;
class Evaluator;

class UniformGrid
{
public:
    float _NEIGHBOR_FACTOR = 4.0f;
    float _SMOOTH_FACTOR = 2.0f;
    float _SPLASH_FACTOR = 1.5f;
    float _ISO_VALUE = 0.0;

    float _MAX_SCALAR = 0.0;
    float _MIN_SCALAR = 0.0;

    bool _USE_ANI = true;
    bool _USE_XMEAN = true;
    float _XMEAN_DELTA = 0.0;
    
    std::shared_ptr<HashGrid> _hashgrid;
    std::shared_ptr<MultiLevelSearcher> _searcher;
    std::shared_ptr<Evaluator> _evaluator;

    std::vector<Eigen::Vector3f> _GlobalParticles;
    std::vector<float> _GlobalRadiuses;

    float _RADIUS = 0;
    int _GlobalParticlesNum = 0;
    float _BoundingBox[6] = {0.0f};
    int dims[3] = {0};
    int steps[3] = {0};

    std::vector<float> _Scalars;
    std::vector<Eigen::Vector3f> _gradients;
    
    void loadRootBox() ;

    void resizeRootBoxConstR();

    void resizeRootBoxVarR();

    void gridSampling();

    UniformGrid() {};

    UniformGrid(
        const std::vector<Eigen::Vector3f>& particles,
        const std::vector<float>& radiuses);

    UniformGrid(
        const std::vector<Eigen::Vector3f>& particles,
        float radius);

    ~UniformGrid() {};

    void Run(float iso_value, std::string filename, std::string filepath);
};

#endif