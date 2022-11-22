#include <float.h>
#include "evaluator.h"
#include "surface_reconstructor.h"


Evaluator::Evaluator(SurfReconstructor* surf_constructor,
		std::vector<Eigen::Vector3f>* global_particles, std::vector<float>* global_density, std::vector<float>* global_mass)
{
    constructor = surf_constructor;
	GlobalPoses = global_particles;
	GlobalDensity = global_density;
	GlobalMass = global_mass;
    SurfaceNormals.clear();
    GlobalSplash.reserve(constructor->getGlobalParticlesNum());

    GlobalxMeans = new Eigen::Vector3f[constructor->getGlobalParticlesNum()];
    GlobalGs = new Eigen::Matrix3f[constructor->getGlobalParticlesNum()];
    sample_step = constructor->getPRadius();
    influnce2 = constructor->getInfluence() * constructor->getInfluence();

	if (constructor->getUseAni())
	{
        compute_Gs_xMeans();
        GlobalSplash.shrink_to_fit();
        printf("   Splash number = %d\n", GlobalSplash.size());
	}
}

void Evaluator::SingleEval(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient, bool use_normalize, bool use_signed, bool grad_normalize)
{
	if (constructor->getMaxScalar() >= 0)
	{
		scalar = 0;
        if (!gradient.isZero())
        {
            gradient.setZero();
        }
	}

	float info = 0.0f;
	float temp_scalars[6] = {0.0f};

    if (constructor->getUseAni())
    {
        AnisotropicEval(pos, info, temp_scalars);
    }
    else
    {
        IsotropicEval(pos, info, temp_scalars);
    }
	
	if (constructor->getMaxScalar() < 0)
	{
		scalar = std::isnan(info) ? 255.0 : info;
		return;
	}
	else
	{
        if (use_normalize)
        {
            scalar = ((info - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
        }
        if (use_signed)
        {
            scalar = constructor->getIsoValue() - scalar;
        }
		gradient[0] = ((temp_scalars[1] - temp_scalars[0]) / constructor->getMaxScalar() * 255) / (constructor->getPRadius() * 2);
		gradient[1] = ((temp_scalars[3] - temp_scalars[2]) / constructor->getMaxScalar() * 255) / (constructor->getPRadius() * 2);
		gradient[2] = ((temp_scalars[5] - temp_scalars[4]) / constructor->getMaxScalar() * 255) / (constructor->getPRadius() * 2);
        if (grad_normalize)
        {
            gradient.normalize();
            gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
            gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
            gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
        }
	}
}

void Evaluator::GridEval(
    std::vector<Eigen::Vector3f>& sample_points, std::vector<float>& field_scalars, std::vector<Eigen::Vector3f>& field_gradients,
    bool& signchange, int oversample)
{
    assert(!sample_points.empty());
    bool origin_sign;
    for (Eigen::Vector3f p : sample_points)
    {
        float scalar = 0.0f;
        if (constructor->getUseAni())
        {
            std::vector<int> pIdxList;
            constructor->getHashGrid()->GetPIdxList(p, pIdxList);
            if (!pIdxList.empty())
            {
                for (int pIdx : pIdxList)
                {
                    scalar += AnisotropicInterpolate(pIdx, p - GlobalxMeans[pIdx]);
                }
            }
        }
        else
        {
            std::vector<int> pIdxList;
            constructor->getHashGrid()->GetPIdxList(p, pIdxList);
            if (!pIdxList.empty())
            {
                float d;
                for (int pIdx : pIdxList)
                {
                    d = (p - GlobalPoses->at(pIdx)).norm();
                    scalar += IsotrpicInterpolate(pIdx, d);
                }
            }
        }
        scalar = constructor->getIsoValue() - ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
        field_scalars.push_back(scalar);
        origin_sign = field_scalars[0] >= 0;
        if (!signchange)
        {
            signchange = origin_sign ^ (scalar >= 0);
        }
    }
    int index, next_idx, last_idx;
    for (int z = 0; z <= oversample; z++)
    {
        for (int y = 0; y <= oversample; y++)
        {
            for (int x = 0; x <= oversample; x++)
            {
                index = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                Eigen::Vector3f gradient(0.0f, 0.0f, 0.0f);

                next_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x + 1));
                last_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x - 1));
                if (x == 0)
                {
                    gradient[0] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (x == oversample)
                {
                    gradient[0] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[0] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                next_idx = (z * (oversample + 1) * (oversample + 1) + (y + 1) * (oversample + 1) + x);
                last_idx = (z * (oversample + 1) * (oversample + 1) + (y - 1) * (oversample + 1) + x);
                if (y == 0)
                {
                    gradient[1] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (y == oversample)
                {
                    gradient[1] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[1] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                next_idx = ((z + 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                last_idx = ((z - 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                if (z == 0)
                {
                    gradient[2] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (z == oversample)
                {
                    gradient[2] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[2] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                gradient.normalize();
                gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
                gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
                gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
                field_gradients.push_back(gradient);
            }
        }
    }
}

bool Evaluator::CheckSplash(const int& pIdx)
{
    if (std::find(GlobalSplash.begin(), GlobalSplash.end(), pIdx) != GlobalSplash.end())
    {
        return true;
    }
    return false;
}

float Evaluator::RecommendIsoValue()
{
    double k_value;
    double recommend_dist = constructor->getPRadius() * 1.0;
    switch (constructor->getKernelType())
    {
    case 0:
        k_value = general_kernel(recommend_dist * recommend_dist, influnce2);
        break;
    case 1:
        k_value = spiky_kernel(recommend_dist, constructor->getInfluence());
        break;
    case 2:
        k_value = viscosity_kernel(recommend_dist, constructor->getInfluence());
        break;
    default:
        k_value = general_kernel(recommend_dist * recommend_dist, influnce2);
        break;
    }
    float scalar = (GlobalMass->at(0) / (*std::min_element(GlobalDensity->begin(), GlobalDensity->end())) * k_value);
    return ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

float Evaluator::RecommendSurfaceThreshold()
{
    double inside = 0, outside = 0;
    const double radius = constructor->getPRadius();
    const double radius2 = radius * radius;
    const double sqrt3 = 1.7320508075688772935274463415059;
    switch (constructor->getKernelType())
    {
    case 0:
        inside += general_kernel(radius2, influnce2);
        inside += general_kernel((5 - 2 * sqrt3) * radius2, influnce2) * 2;
        //inside += general_kernel(5 * radius2, influnce2) * 4;
        outside += general_kernel(radius2, influnce2);
        //outside += general_kernel(5 * radius2, influnce2) * 2;
        outside += general_kernel((5 + 2 * sqrt3) * radius2, influnce2) * 2;
        //outside += general_kernel(13 * radius2, influnce2) * 2;
        break;
    case 1:
        inside += spiky_kernel(radius, constructor->getInfluence());
        inside += spiky_kernel(sqrt(5) * radius, constructor->getInfluence());
        outside += spiky_kernel(radius, influnce2);
        outside += spiky_kernel(sqrt(5) * radius, influnce2);
        outside += spiky_kernel(sqrt(9) * radius, influnce2);
        outside += spiky_kernel(sqrt(13) * radius, influnce2);
        break;
    case 2:
        inside += viscosity_kernel(radius, constructor->getInfluence());
        inside += viscosity_kernel(sqrt(5) * radius, constructor->getInfluence());
        outside += viscosity_kernel(radius, influnce2);
        outside += viscosity_kernel(sqrt(5) * radius, influnce2);
        outside += viscosity_kernel(sqrt(9) * radius, influnce2);
        outside += viscosity_kernel(sqrt(13) * radius, influnce2);
        break;
    default:
        inside += general_kernel(radius2, influnce2);
        inside += general_kernel(5 * radius2, influnce2);
        outside += general_kernel(radius2, influnce2);
        outside += general_kernel(5 * radius2, influnce2);
        outside += general_kernel(9 * radius2, influnce2);
        outside += general_kernel(13 * radius2, influnce2);
        break;
    }
    inside *= (GlobalMass->at(0) / 1000);
    outside *= (GlobalMass->at(0) / 1000);
    return (std::abs(((outside - inside) / constructor->getMaxScalar() * 255) / (radius * 2)));
}

void Evaluator::GetSurfaceParticles()
{
    float recommand_surface_threshold = RecommendSurfaceThreshold();
    printf("   Recommend Surface Threshold = %f\n", recommand_surface_threshold);
#pragma omp parallel for schedule(static, OMP_THREADS_NUM) 
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        float tempScalar = 0;
        Eigen::Vector3f tempGrad = Eigen::Vector3f::Zero();
        if (!CheckSplash(pIdx))
        {
            if (SurfaceNormals.find(pIdx) != SurfaceNormals.end())
            {
                //if (SurfaceNormals[pIdx] == Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX))
                //{
                //    continue;
                //} else {
                    SingleEval(GlobalPoses->at(pIdx), tempScalar, tempGrad);
                    SurfaceNormals[pIdx][0] = tempGrad[0];
                    SurfaceNormals[pIdx][1] = tempGrad[1];
                    SurfaceNormals[pIdx][2] = tempGrad[2];
                //}
            } else {
                SingleEval(GlobalPoses->at(pIdx), tempScalar, tempGrad, true, true, false);
                if (std::abs(tempGrad.maxCoeff()) > recommand_surface_threshold || std::abs(tempGrad.minCoeff()) > recommand_surface_threshold || tempGrad.norm() > recommand_surface_threshold)
                {
                    SurfaceNormals[pIdx] = Eigen::Vector3f(tempGrad.normalized());
                }
            }
        }
    }
    printf("   Surface particles number = %d\n", SurfaceNormals.size());
}

inline float Evaluator::general_kernel(double d2, double h2)
{
    double p_dist = (d2 >= h2 ? 0.0 : pow(h2 - d2, 3));
    double sigma = (315 / (64 * pow(constructor->getInfluence(), 9))) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::spiky_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : pow(h - d, 3));
    double sigma = (15 / pow(constructor->getInfluence(), 6)) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::viscosity_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : (-pow(d, 3) / (2 * pow(h, 3)) + pow(d, 2) / pow(h, 2) + d / (2 * h) - 1));
    double sigma = (15 / (2 * pow(h, 3))) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::IsotrpicInterpolate(const int pIdx, const float d)
{
	if (d > constructor->getInfluence())
		return 0.0f;
	float kernel_value = 315 / (64 * pow(constructor->getInfluence(), 9)) * 0.318309886183790671538 * pow(((constructor->getInfluence() * constructor->getInfluence()) - (d * d)), 3);
	return (*GlobalMass)[pIdx] / (*GlobalDensity)[pIdx] * kernel_value;
}

inline float Evaluator::AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff)
{
    double k_value;
    switch (constructor->getKernelType())
    {
    case 0:
        k_value = general_kernel((GlobalGs[pIdx] * diff).squaredNorm(), influnce2);
        break;
    case 1:
        k_value = spiky_kernel((GlobalGs[pIdx] * diff).norm(), constructor->getInfluence());
        break;
    case 2:
        k_value = viscosity_kernel((GlobalGs[pIdx] * diff).norm(), constructor->getInfluence());
        break;
    default:
        k_value = general_kernel((GlobalGs[pIdx] * diff).squaredNorm(), influnce2);
        break;
    }
    return (GlobalMass->at(pIdx) / GlobalDensity->at(pIdx)) * (GlobalGs[pIdx].determinant() * k_value);
}

inline void Evaluator::compute_Gs_xMeans()
{
	const double h = constructor->getPRadius();
	const double h2 = h * h;
	const double R = constructor->getInfluence();
    const double R2 = R * R;
    const double invH = 1.0;

#pragma omp parallel for //schedule(static, 16) 
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        std::vector<int> tempNeighborList;
        std::vector<int> neighborList;
        Eigen::Vector3f xMean = Eigen::Vector3f::Zero();
        Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
        constructor->getHashGrid()->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighborList);
        if (tempNeighborList.empty())
            continue;
        double wSum = 0, d2, d, wj;
        for (int nIdx : tempNeighborList)
        {
            if (nIdx == pIdx)
                continue;
            d2 = (((GlobalPoses->at(nIdx))) - ((GlobalPoses->at(pIdx)))).squaredNorm();
            if (d2 >= R2)
            {
                continue;
            }
            d = sqrt(d2);
            wj = wij(d, R);
            wSum += wj;
            xMean += ((GlobalPoses->at(nIdx))) * wj;
            neighborList.push_back(nIdx);
        }
        if (constructor->getUseXMean())
        {
            if (wSum > 0)
            {
                xMean /= wSum;
                xMean = (GlobalPoses->at(pIdx)) * (1 - constructor->getXMeanDelta()) + xMean * constructor->getXMeanDelta();
            }
            else
            {
                xMean = (GlobalPoses->at(pIdx));
            }
        }
        else
        {
            xMean = (GlobalPoses->at(pIdx));
        }

        if (neighborList.size() < 1)
        {
            G += Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
            GlobalSplash.push_back(pIdx);
        } 
        else
        {
            if (neighborList.size() < constructor->getMinNeighborsNum())
            {
                SurfaceNormals[pIdx] = Eigen::Vector3f(0.0, 0.0, 0.0);
                //if (neighborList.size() < constructor->getSparseNeighborsNum())
                //{
                //    SurfaceNormals[pIdx] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
                //}
            }
            Eigen::Vector3f wd = Eigen::Vector3f::Zero();
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            cov += Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
            wSum = 0.0f;
            for (int nIdx : neighborList)
            {
                d = (xMean - ((GlobalPoses->at(nIdx)))).norm();
                wj = wij(d, R);
                wSum += wj;
                wd = ((GlobalPoses->at(nIdx))) - xMean;
                cov += ((wd * wd.transpose()).cast<double>() * wj);
            }
            cov /= wSum;

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            
            Eigen::Matrix3d u = svd.matrixU();
            Eigen::Vector3d w = svd.singularValues();
            Eigen::Matrix3d v = svd.matrixV();
            
            w = Eigen::Vector3d(w.array().abs());
            const double maxSingularVal = w.maxCoeff();

            const double kr = 4.0;
            w[0] = std::max(w[0], maxSingularVal / kr);
            w[1] = std::max(w[1], maxSingularVal / kr);
            w[2] = std::max(w[2], maxSingularVal / kr);

            Eigen::Matrix3d invSigma = w.asDiagonal().inverse();
            // Compute G
            const double scale =
                std::pow(w[0] * w[1] * w[2], 1.0 / 3.0);  // volume preservation
            G = ((v * invSigma * u.transpose()) * invH * scale).cast<float>();
        }
        GlobalxMeans[pIdx] = Eigen::Vector3f(xMean);
        GlobalGs[pIdx] = Eigen::Matrix3f(G);
    }
}

inline double Evaluator::wij(double d, double r)
{
	if (d < r)
	{
		return (1.0 - pow(d / r, 3));
	}
	else
	{
		return 0.0;
	}
}

inline void Evaluator::IsotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars)
{
    Eigen::Vector3f x_up_pos(pos[0] + sample_step, pos[1], pos[2]),
        x_down_pos(pos[0] - sample_step, pos[1], pos[2]),
        y_up_pos(pos[0], pos[1] + sample_step, pos[2]),
        y_down_pos(pos[0], pos[1] - sample_step, pos[2]),
        z_up_pos(pos[0], pos[1], pos[2] + sample_step),
        z_down_pos(pos[0], pos[1], pos[2] - sample_step);
    
    std::vector<int> pIdxList;
    constructor->getHashGrid()->GetPIdxList(pos, pIdxList);
    if (pIdxList.empty())
    {
        return;
    }
    float d;
    for (int pIdx : pIdxList)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }
        
        d = (pos - GlobalPoses->at(pIdx)).norm();
        info += IsotrpicInterpolate(pIdx, d);

        d = (x_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[0] += IsotrpicInterpolate(pIdx, d);

        d = (x_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[1] += IsotrpicInterpolate(pIdx, d);

        d = (y_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[2] += IsotrpicInterpolate(pIdx, d);

        d = (y_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[3] += IsotrpicInterpolate(pIdx, d);

        d = (z_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[4] += IsotrpicInterpolate(pIdx, d);

        d = (z_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[5] += IsotrpicInterpolate(pIdx, d);
    }
}

inline void Evaluator::AnisotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars)
{
    Eigen::Vector3f x_up_pos(pos[0] + sample_step, pos[1], pos[2]),
        x_down_pos(pos[0] - sample_step, pos[1], pos[2]),
        y_up_pos(pos[0], pos[1] + sample_step, pos[2]),
        y_down_pos(pos[0], pos[1] - sample_step, pos[2]),
        z_up_pos(pos[0], pos[1], pos[2] + sample_step),
        z_down_pos(pos[0], pos[1], pos[2] - sample_step);

    std::vector<int> pIdxList;
    constructor->getHashGrid()->GetPIdxList(pos, pIdxList);
    if (pIdxList.empty())
    {
        return;
    }
    Eigen::Vector3f diff;
    for (int pIdx : pIdxList)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }
        
        diff = pos - GlobalxMeans[pIdx];
        info += AnisotropicInterpolate(pIdx, diff);

        diff = x_up_pos - GlobalxMeans[pIdx];
        temp_scalars[0] += AnisotropicInterpolate(pIdx, diff);

        diff = x_down_pos - GlobalxMeans[pIdx];
        temp_scalars[1] += AnisotropicInterpolate(pIdx, diff);

        diff = y_up_pos - GlobalxMeans[pIdx];
        temp_scalars[2] += AnisotropicInterpolate(pIdx, diff);

        diff = y_down_pos - GlobalxMeans[pIdx];
        temp_scalars[3] += AnisotropicInterpolate(pIdx, diff);

        diff = z_up_pos - GlobalxMeans[pIdx];
        temp_scalars[4] += AnisotropicInterpolate(pIdx, diff);

        diff = z_down_pos - GlobalxMeans[pIdx];
        temp_scalars[5] += AnisotropicInterpolate(pIdx, diff);
    }
}
