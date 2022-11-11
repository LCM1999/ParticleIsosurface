#pragma once
#ifndef SURF_CONSTRUCT_H
#define SURF_CONSTRUCT_H
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <metis.h>
#include "timer.h"
#include "utils.h"

class HashGrid;
class Evaluator;
class TNode;
class Mesh;

class SurfReconstructor
{
private:
    // Global Parameters
    int _OVERSAMPLE_QEF = 4;
    float _BORDER = (1.0 / 4096.0);
    int _DEPTH_MAX = 6; // 7
    int _DEPTH_MIN = 5; // 4
    float _P_RADIUS = 0.0f;	// 0.00025f;
    float _INFLUENCE = 0.0f;
    float _ISO_VALUE = 0.0f;

    int _KERNEL_TYPE = 0;

    float _MAX_SCALAR = -1.0f;
    float _MIN_SCALAR = 0.0f;
    float _BAD_QEF = 0.0001f;
    float _FLATNESS = 0.99f;
    int _MIN_NEIGHBORS_NUM = 5;

    bool _USE_ANI = true;
    bool _USE_XMEAN = true;
    float _XMEAN_DELTA = 0.3;

    float _TOLERANCE = 1e-8;
    float _RATIO_TOLERANCE = 0.1f;
    //float _LOW_MESH_QUALITY = 0.17320508075688772935274463415059;
    float _MESH_TOLERANCE = 1e4;

    HashGrid* _hashgrid = NULL;
    Evaluator* _evaluator = NULL;

    std::vector<Eigen::Vector3f> _GlobalParticles;
    std::vector<float> _GlobalDensity;
    std::vector<float> _GlobalMass;

    int _GlobalParticlesNum = 0;
    int _STATE = 0;

    double _BoundingBox[6] = {0.0f};
    double _RootHalfLength;
    float _RootCenter[3] = {0.0f};

    TNode* _OurRoot;
	Mesh* _OurMesh;
protected:
    void loadRootBox();

    void generalModeRun();

    void resizeRootBox();

    // Method for CSV mode
    void genIsoOurs();
    void eval(TNode* tnode, Eigen::Vector3f* grad, TNode* guide);
    void recordProgress(TNode* root_node, const char* record_name);

public:
    SurfReconstructor() {};
    SurfReconstructor(std::vector<Eigen::Vector3f>& particles, std::vector<float>& density, std::vector<float>& mass, Mesh& mesh, float particle_radius);

    ~SurfReconstructor() {};

    void Run();

    inline float getOverSampleQEF() {return _OVERSAMPLE_QEF;};
    inline float getBorder() {return _BORDER;};
    inline int getDepthMax() {return _DEPTH_MAX;}
    inline int getDepthMin() {return _DEPTH_MIN;}
    inline float getPRadius() {return _P_RADIUS;}
    inline float getInfluence() {return _INFLUENCE;}
    inline float getIsoValue() {return _ISO_VALUE;}
    inline int getKernelType() {return _KERNEL_TYPE;}
    inline float getMaxScalar() {return _MAX_SCALAR;}
    inline float getMinScalar() {return _MIN_SCALAR;}
    inline float getBadQef() {return _BAD_QEF;}
    inline float getFlatness() {return _FLATNESS;}
    inline int getMinNeighborsNum() {return _MIN_NEIGHBORS_NUM;}
    inline bool getUseAni() {return _USE_ANI;}
    inline bool getUseXMean() {return _USE_XMEAN;}
    inline float getXMeanDelta() {return _XMEAN_DELTA;}
    inline float getTolerance() {return _TOLERANCE;}
    inline float getRatioTolerance() {return _RATIO_TOLERANCE;}
    inline float getMeshTolerance() {return _MESH_TOLERANCE;}
    //inline int getCsvType() {return _CSV_TYPE;}
    //inline std::string getOutputPrefix() {return _OUTPUT_PREFIX;}
    //inline std::string getRecordPrefix() {return _RECORD_PREFIX;}
    //inline bool getLoadRecord() {return _LOAD_RECORD;}
    //inline bool getNeedRecord() {return _NEED_RECORD;}
    //inline int getRecordStep() {return _RECORD_STEP;}
    inline HashGrid* getHashGrid() {return _hashgrid;}
    inline Evaluator* getEvaluator() {return _evaluator;}
    inline int getGlobalParticlesNum() {return _GlobalParticlesNum;}
    //inline int getINDEX() {return _INDEX;}
    inline int getSTATE() {return _STATE;}
};

#endif
