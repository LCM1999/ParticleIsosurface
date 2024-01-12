#pragma once

#include "iso_common.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

class SurfReconstructor;

class Evaluator
{
public:
	Evaluator() {};
	~Evaluator() {};

    SurfReconstructor* constructor;
	std::vector<Eigen::Vector3f>* GlobalPoses;
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
    std::vector<std::uint8_t> GlobalSplash;
    std::vector<Eigen::Vector3f> PariclesNormals;
	std::vector<Eigen::Vector3f> GlobalxMeans;
    std::vector<Eigen::Matrix3f> GlobalGs;
    std::vector<float> GlobalDeterminant;

	Evaluator(  SurfReconstructor* surf_constructor, std::vector<Eigen::Vector3f>* global_particles, 
                std::vector<float>* radiuses,
                float radius);

	void SingleEval(const Eigen::Vector3f& pos, float& scalar);
    void SingleEvalWithGrad(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient);
    void GridEval(
        float* sample_points, float* field_gradients, float cellsize, 
        bool& signchange, int oversample, bool grad_normalize = false);

    bool CheckSplash(const int& pIdx);
    float CalculateMaxScalarConstR();
    float CalculateMaxScalarVarR();
    float RecommendIsoValueConstR(const float iso_factor);
    float RecommendIsoValueVarR(const float iso_factor);
    void CalcParticlesNormal();
private:
    const float sqrt3 = 1.7320508075688772935274463415059;
    const float sqrt11 = 3.3166247903553998491149327366707;
    const float inv_pi = 0.31830988618379067153776752674503;

    float NeighborFactor;
    float SmoothFactor;

    float general_kernel(float d2, float h2, float sigma);
    Eigen::Vector3f gradient_kernel(float d2, float h2, float sigma, Eigen::Vector3f diff);

	float IsotropicInterpolate(const int pIdx, const float d);
    Eigen::Vector3f IsotropicInterpolateGrad(const int pIdx, const float d2, const Eigen::Vector3f diff);
	float AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f diff);
    Eigen::Vector3f AnisotropicInterpolateGrad(const int pIdx, const Eigen::Vector3f diff);
	void compute_Gs_xMeans();
    void compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3f &xMean);
    void compute_G(Eigen::Vector3f p, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G);
	float wij(float d, float r);
};
