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
	Evaluator();
	~Evaluator();

    SurfReconstructor* constructor;
	std::vector<Eigen::Vector3f>* GlobalPoses;
    std::vector<float>* GlobalRadius;
    float _MAX_RADIUS, _MIN_RADIUS;

    float Radius = 0;
    std::vector<bool> GlobalSplash;
    std::vector<Eigen::Vector3f> PariclesNormals;
	Eigen::Vector3f* GlobalxMeans;
    Eigen::Matrix3f* GlobalGs;

	Evaluator(  SurfReconstructor* surf_constructor, std::vector<Eigen::Vector3f>* global_particles, 
                std::vector<float>* radiuses,
                float radius);

	void SingleEval(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient, bool use_normalize = true, bool use_signed = true, bool grad_normalize = true);

    void GridEval(
        float* sample_points, float* field_gradients,
        bool& signchange, int oversample, bool grad_normalize = true);

    bool CheckSplash(const int& pIdx);
    float CalculateMaxScalarConstR();
    float CalculateMaxScalarVarR();
    float RecommendIsoValueConstR();
    float RecommendIsoValueVarR();
    void CalcParticlesNormal();
private:
    const double bv_factor = 4.1887902047863909846168578443727;
    const double inv_pi = 0.31830988618379067153776752674503;

    float inf_factor;

    float general_kernel(double d2, double h2, double h);
    float spiky_kernel(double d, double h);
    float viscosity_kernel(double d, double h);

	float IsotropicInterpolate(const int pIdx, const double d);
	float AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff);
	void compute_Gs_xMeans();
	double wij(double d, double r);

    void IsotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius);
    void AnisotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius);
};
