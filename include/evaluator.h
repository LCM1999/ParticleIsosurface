#pragma once

#include "iso_common.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>


class Evaluator
{
private:
    const float sqrt2 = 1.4142135623730950488016887242097;
    const float sqrt3 = 1.7320508075688772935274463415059;
    const float sqrt11 = 3.3166247903553998491149327366707;
    const float inv_pi = 0.31830988618379067153776752674503;

    float _NEIGHBOR_FACTOR = 4.0;
    float _SMOOTH_FACTOR = 2.0;
    float _ISO_FACTOR = 1.9;
    float _ISO_VALUE = 0.0;

    float _MAX_SCALAR = -1.0;
    float _MIN_SCALAR = 0.0;

    bool _USE_XMEAN = true;
    float _XMEAN_DELTA = 0.0;

    float M = 1.05f;

    float poly6_kernel(float d2, float h2, float sigma);
    float Bspline_kernel(float ratio, float sigma);
    float Gaussian_kernel(float ratio2, float sigma);
    Eigen::Vector3f poly6_gradient_kernel(float d2, float h2, float sigma, Eigen::Vector3f diff);
    Eigen::Vector3f Bspline_gradient_kernel(float ratio, float sigma, Eigen::Vector3f diff);

	float IsotropicInterpolate(const int pIdx, const float d);
    Eigen::Vector3f IsotropicInterpolateGrad(const int pIdx, const float d2, const Eigen::Vector3f diff);
	float AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f diff);
    Eigen::Vector3f AnisotropicInterpolateGrad(const int pIdx, const Eigen::Vector3f diff);
    void compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3f &xMean);
    void compute_G_ours(int pIdx, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G);
    void compute_G_Yus(int pIdx, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G);
	float wij(float d, float r);

public:
	Evaluator() {};
	~Evaluator() {};

    std::shared_ptr<HashGrid> _hashgrid;
    std::shared_ptr<MultiLevelSearcher> _searcher;

	std::vector<Eigen::Vector3f>* GlobalPoses;
    int _GlobalParticlesNum = 0;
    std::vector<float>* GlobalRadius;
    std::vector<float> GlobalRadius2;
    std::vector<float> GlobalRadius3;
    std::vector<float> GlobalInflunce2;
    std::vector<float> GlobalSigma;

    float Radius = 0;
    float Radius2 = 0;
    float Radius3 = 0;
    float Influnce2 = 0;
    float Sigma = 0;
    std::vector<bool> GlobalSplash;
    std::vector<bool> GlobalSurface;
    std::vector<Eigen::Vector3f> PariclesNormals;
	std::vector<Eigen::Vector3f> GlobalxMeans;
    std::vector<Eigen::Matrix3f> GlobalGs;
    std::vector<float> GlobalDeterminant;

	Evaluator(  std::shared_ptr<HashGrid>& hashgrid,
                std::shared_ptr<MultiLevelSearcher>& searcher,
                std::vector<Eigen::Vector3f>* global_particles, 
                std::vector<float>* radiuses,
                float radius);

	void SingleEval(const Eigen::Vector3f& pos, float& scalar);
    void SingleEvalWithGrad(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient);
    void GridEval(
        float* sample_points, float* field_gradients, float cellsize, 
        bool& signchange, int oversample, bool grad_normalize = false);

    bool CheckSplash(const int& pIdx);
    bool CheckSurface(const int& pIdx);
    void CalculateMaxScalarConstR();
    void CalculateMaxScalarVarR();
    void RecommendIsoValueConstR();
    void RecommendIsoValueVarR();
    void CalcParticlesNormal();
	void compute_Gs_xMeans();

    inline float getNeighborFactor() {return _NEIGHBOR_FACTOR;}
    inline float getSmoothFactor() {return _SMOOTH_FACTOR;}
    inline void setSmoothFactor(float smooth_factor) {_SMOOTH_FACTOR = smooth_factor;}
    inline float getIsoFactor() {return _ISO_FACTOR;}
    inline void setIsoFactor(float iso_factor) {_ISO_FACTOR = iso_factor;}
    inline float getIsoValue() {return _ISO_VALUE;}
    inline void setIsoValue(float iso_value) {_ISO_VALUE = iso_value;}
    inline float getMaxScalar() {return _MAX_SCALAR;}
    inline float getMinScalar() {return _MIN_SCALAR;}
    inline bool getUseXMean() {return _USE_XMEAN;}
    inline float getXMeanDelta() {return _XMEAN_DELTA;}
};
