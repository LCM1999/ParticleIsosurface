#include <float.h>
#include "evaluator.h"
#include "surface_reconstructor.h"


Evaluator::Evaluator(SurfReconstructor* surf_constructor,
		std::vector<Eigen::Vector3f>* global_particles, 
        std::vector<float>* radiuses, float radius)
{
    constructor = surf_constructor;
	GlobalPoses = global_particles;
    GlobalRadius = radiuses;
    if (!IS_CONST_RADIUS)
    {
        _MAX_RADIUS = constructor->getSearcher()->getMaxRadius();
        _MIN_RADIUS = constructor->getSearcher()->getMinRadius();
    }
    Radius = radius;

    neighbor_factor = constructor->getNeighborFactor();
    smooth_factor = constructor->getSmoothFactor();


    PariclesNormals.clear();
    GlobalSplash.clear();
    PariclesNormals.resize(constructor->getGlobalParticlesNum(), Eigen::Vector3f(0, 0, 0));
    GlobalSplash.resize(constructor->getGlobalParticlesNum(), 0);

    GlobalxMeans = new Eigen::Vector3f[constructor->getGlobalParticlesNum()];
    GlobalGs = new Eigen::Matrix3f[constructor->getGlobalParticlesNum()];
	if (constructor->getUseAni())
	{
        compute_Gs_xMeans();
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
    float sample_radius = 0.0f;

    if (constructor->getUseAni())
    {
        AnisotropicEval(pos, info, temp_scalars, sample_radius);
    }
    else
    {
        IsotropicEval(pos, info, temp_scalars, sample_radius);
    }
	
    if (use_normalize)
    {
        scalar = ((info - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
    }
    if (use_signed)
    {
        scalar = constructor->getIsoValue() - scalar;
    }
	gradient[0] = ((temp_scalars[1] - temp_scalars[0]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
	gradient[1] = ((temp_scalars[3] - temp_scalars[2]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
	gradient[2] = ((temp_scalars[5] - temp_scalars[4]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
    if (grad_normalize)
    {
        gradient.normalize();
        gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
        gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
        gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
    }
}

void Evaluator::GridEval(
    float* sample_points, float* field_gradients,
    bool& signchange, int oversample, bool grad_normalize)
{
    //assert(!sample_points.empty());
    bool origin_sign;
    for (int i = 0; i < pow(oversample+1, 3); i++)
    {
        Eigen::Vector4f p(sample_points[i*4 + 0], sample_points[i*4 + 1], sample_points[i*4 + 2], sample_points[i*4 + 3]);
        float scalar = 0.0f;
        if (constructor->getUseAni())
        {
            std::vector<int> neighbors;
            if (IS_CONST_RADIUS)
            {
                constructor->getHashGrid()->GetPIdxList(p.head(3), neighbors);
            } else {
                constructor->getSearcher()->GetNeighbors(p.head(3), neighbors);
            }
            if (!neighbors.empty())
            {
                for (const int pIdx : neighbors)
                {
                    scalar += AnisotropicInterpolate(pIdx, p.head(3) - GlobalxMeans[pIdx]);
                }
            }
        }
        else
        {
            std::vector<int> neighbors;
            if (IS_CONST_RADIUS)
            {
                constructor->getHashGrid()->GetPIdxList(p.head(3), neighbors);
            } else {
                constructor->getSearcher()->GetNeighbors(p.head(3), neighbors);
            }
            if (!neighbors.empty())
            {
                double d2;
                for (int pIdx : neighbors)
                {
                    d2 = (p.head(3) - GlobalPoses->at(pIdx)).squaredNorm();
                    scalar += IsotropicInterpolate(pIdx, d2);
                }
            }
        }
        scalar = constructor->getIsoValue() - ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
        sample_points[i*4 + 3] = scalar;
        origin_sign = (sample_points[3] >= 0);
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
                    gradient[0] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (x == oversample)
                {
                    gradient[0] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[0] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
                }

                next_idx = (z * (oversample + 1) * (oversample + 1) + (y + 1) * (oversample + 1) + x);
                last_idx = (z * (oversample + 1) * (oversample + 1) + (y - 1) * (oversample + 1) + x);
                if (y == 0)
                {
                    gradient[1] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (y == oversample)
                {
                    gradient[1] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[1] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
                }

                next_idx = ((z + 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                last_idx = ((z - 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                if (z == 0)
                {
                    gradient[2] = sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                else if (z == oversample)
                {
                    gradient[2] = sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3];
                }
                else
                {
                    gradient[2] = sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3];
                }
                if (grad_normalize)
                {
                    gradient.normalize();
                    gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
                    gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
                    gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
                }
                field_gradients[index * 3 + 0] = gradient[0];
                field_gradients[index * 3 + 1] = gradient[1];
                field_gradients[index * 3 + 2] = gradient[2];
            }
        }
    }
}

bool Evaluator::CheckSplash(const int& pIdx)
{
    if (GlobalSplash[pIdx])
    {
        return true;
    }
    return false;
}

float Evaluator::CalculateMaxScalarConstR()
{
    double k_value = 0;
    const double radius = Radius;
    const double radius2 = radius * radius;
    const double influnce = radius * smooth_factor;
    const double influnce2 = influnce * influnce;

    k_value += general_kernel(3 * radius2, influnce2, influnce) * 8;
    k_value += general_kernel(11 * radius2, influnce2, influnce) * 4 * 6;

    return radius * radius2 * k_value;
}

float Evaluator::CalculateMaxScalarVarR()
{
    double max_scalar = 0;
    std::vector<float>* radiuses = constructor->getSearcher()->getCheckedRadiuses();

    double radius2, influnce, influnce2;
    for (const float r : *radiuses)
    {
        double temp_scalar = 0;
        radius2 = r * r;
        influnce = r * smooth_factor;
        influnce2 = influnce * influnce;
        temp_scalar += general_kernel(3 * radius2, influnce2, influnce) * 8;
        temp_scalar += general_kernel(11 * radius2, influnce2, influnce) * 4 * 6;
        temp_scalar = r * radius2 * temp_scalar;
        if (temp_scalar > max_scalar)
        {
            max_scalar = temp_scalar;
        }
    }
    return max_scalar;
}

float Evaluator::RecommendIsoValueConstR()
{
    double k_value = 0.0;
    const double radius = Radius;
    const double radius2 = radius * radius;

    const double influnce = radius * smooth_factor;
    const double influnce2 = influnce * influnce;

    // const double invH = 1.0;
    // double wSum = 0, d, wj;
    // double nR, nI;
    // Eigen::Vector3f p1(radius, 0, 0);
    // Eigen::Vector3f p2(-radius, 0, 0);
    // Eigen::Vector3f sample(radius, radius,0);
    // Eigen::Matrix3f G1 = Eigen::Matrix3f::Zero();
    // Eigen::Matrix3f G2 = Eigen::Matrix3f::Zero();
    // compute_single_G(p1, p2, G1, radius);
    // compute_single_G(p2, p1, G2, radius);

    // k_value += general_kernel((G1 * (sample - p1)).squaredNorm(), influnce2, influnce);
    // k_value += general_kernel((G2 * (sample - p2)).squaredNorm(), influnce2, influnce);

    k_value += general_kernel(2 * radius2, influnce2, influnce);
    k_value += general_kernel(5 * radius2, influnce2, influnce);


    return (((radius2 * radius * k_value) 
    - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

float Evaluator::RecommendIsoValueVarR()
{
    double recommend = 0.0;
    std::vector<float>* radiuses = constructor->getSearcher()->getCheckedRadiuses();

    double radius2, influnce, influnce2, k_value;
    for (const float r : *radiuses)
    {
        radius2 = r * r;
        influnce = r * smooth_factor;
        influnce2 = influnce * influnce;
        k_value = 0;
        k_value += general_kernel(2 * radius2, influnce2, influnce);
        k_value += general_kernel(5 * radius2, influnce2, influnce);
        recommend += (radius2 * r) * k_value;
    }
    return ((recommend / radiuses->size()) - constructor->getMinScalar()) / constructor->getMaxScalar() * 255;
}

void Evaluator::CalcParticlesNormal()
{
    //float recommand_surface_threshold = RecommendSurfaceThreshold();
    //printf("   Recommend Surface Threshold = %f\n", recommand_surface_threshold);
#pragma omp parallel for schedule(static, OMP_THREADS_NUM) 
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        float tempScalar = 0;
        Eigen::Vector3f tempGrad = Eigen::Vector3f::Zero();
        if (!CheckSplash(pIdx))
        {
            SingleEval(GlobalPoses->at(pIdx), tempScalar, tempGrad);
            PariclesNormals[pIdx][0] = tempGrad[0];
            PariclesNormals[pIdx][1] = tempGrad[1];
            PariclesNormals[pIdx][2] = tempGrad[2];
        }
    }
}

inline float Evaluator::general_kernel(double d2, double h2, double h)
{
    double p_dist = (d2 >= h2 ? 0.0 : pow(h2 - d2, 3));
    double sigma = (315 / (64 * pow(h, 9))) * inv_pi;
    return p_dist * sigma;
}

inline float Evaluator::spiky_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : pow(h - d, 3));
    double sigma = (15 / pow(h, 6)) * inv_pi;
    return p_dist * sigma;
}

inline float Evaluator::viscosity_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : (-pow(d, 3) / (2 * pow(h, 3)) + pow(d, 2) / pow(h, 2) + d / (2 * h) - 1));
    double sigma = (15 / (2 * pow(h, 3))) * inv_pi;
    return p_dist * sigma;
}

inline float Evaluator::IsotropicInterpolate(const int pIdx, const double d2)
{
    double radius = (IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx));
    double influnce = radius * smooth_factor;
    double influnce2 = influnce * influnce;
	if (d2 > influnce2)
		return 0.0f;
	float kernel_value = 315 / (64 * pow(influnce, 9)) * inv_pi * pow((influnce2 - d2), 3);
	return pow(radius, 3) * kernel_value;
}

inline float Evaluator::AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff)
{
    double radius = (IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx));
    double influnce = radius * smooth_factor;
    double influnce2 = influnce * influnce;
    double k_value = general_kernel((GlobalGs[pIdx] * diff).squaredNorm(), influnce2, influnce);
    return (pow(radius, 3)) * (GlobalGs[pIdx].determinant() * k_value);
}

inline void Evaluator::compute_single_G(Eigen::Vector3f p, Eigen::Vector3f neighbor, Eigen::Matrix3f &G, float r)
{
    const double invH = 1.0;
    double wSum = 0, d, wj;

    Eigen::Vector3f wd = Eigen::Vector3f::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    cov += Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
    wSum = 0.0f;
    d = (p - neighbor).norm();
    wj = wij(d, r * neighbor_factor);
    wSum += wj;
    wd = neighbor - p;
    cov += ((wd * wd.transpose()).cast<double>() * wj);
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

inline void Evaluator::compute_G(Eigen::Vector3f p, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G)
{
    const double invH = 1/smooth_factor;
    double wSum = 0, d, wj;
    double nR, nI;

    Eigen::Vector3f wd = Eigen::Vector3f::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    cov += Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
    wSum = 0.0f;
    for (int nIdx : neighbors)
    {
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nI = nR * neighbor_factor;
        d = (p - GlobalPoses->at(nIdx)).norm();
        wj = wij(d, nI);
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

inline void Evaluator::compute_Gs_xMeans()
{
    const double invH = 1.0;
#pragma omp parallel for schedule(static, OMP_THREADS_NUM) 
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        std::vector<int> tempNeighbors;
        std::vector<int> neighbors;
        int closerNeigbors = 0;
        Eigen::Vector3f xMean = Eigen::Vector3f::Zero();
        Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
        double wSum = 0, d2, d, wj;
        double pR, pR2, pD, pD2, pI, pI2;
        pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
        pR2 = pR * pR;
        pD = smooth_factor * pR;
        pD2 = pD * pD;
        pI = pR * neighbor_factor;
        pI2 = pI * pI;
        if (IS_CONST_RADIUS)
        {
            constructor->getHashGrid()->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighbors);
        } else {
            constructor->getSearcher()->GetNeighbors((GlobalPoses->at(pIdx)), tempNeighbors);
        }
        if (tempNeighbors.size() <= 1)
        {
            G = Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
            GlobalSplash[pIdx] = 1;
        }
        double nR, nR2, nD, nD2, nI, nI2;
        for (int nIdx : tempNeighbors)
        {
            if (nIdx == pIdx)
                continue;
            nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
            nR2 = nR * nR;
            nD = smooth_factor * nR;
            nD2 = nD * nD;
            nI = nR * neighbor_factor;
            nI2 = nI * nI;
            d2 = (((GlobalPoses->at(nIdx))) - ((GlobalPoses->at(pIdx)))).squaredNorm();
            if (d2 >= nI2)
            {
                continue;
            }
            d = sqrt(d2);
            wj = wij(d, nI);
            wSum += wj;
            xMean += ((GlobalPoses->at(nIdx))) * wj;
            neighbors.push_back(nIdx);
            if (d2 <= std::max(pD2, nD2) && d2 > 0)
            {
                closerNeigbors++;
            }
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

        if (neighbors.size() < 1)
        {
            G = Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
            GlobalSplash[pIdx] = 1;
        } 
        else
        {
            compute_G(GlobalPoses->at(pIdx), xMean, neighbors, G);
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

inline void Evaluator::IsotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius)
{    
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    if (neighbors.empty())
    {
        return;
    }

    sample_radius = IS_CONST_RADIUS ? Radius : 
    (*std::min_element(neighbors.begin(), neighbors.end(), [&](const int& a, const int& b) {
        return GlobalRadius[a] < GlobalRadius[b];
    }));

    Eigen::Vector3f x_up_pos(pos[0] + sample_radius, pos[1], pos[2]),
                    x_down_pos(pos[0] - sample_radius, pos[1], pos[2]),
                    y_up_pos(pos[0], pos[1] + sample_radius, pos[2]),
                    y_down_pos(pos[0], pos[1] - sample_radius, pos[2]),
                    z_up_pos(pos[0], pos[1], pos[2] + sample_radius),
                    z_down_pos(pos[0], pos[1], pos[2] - sample_radius);

    float d;
    for (int pIdx : neighbors)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }
        
        d = (pos - GlobalPoses->at(pIdx)).norm();
        info += IsotropicInterpolate(pIdx, d);

        d = (x_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[0] += IsotropicInterpolate(pIdx, d);

        d = (x_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[1] += IsotropicInterpolate(pIdx, d);

        d = (y_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[2] += IsotropicInterpolate(pIdx, d);

        d = (y_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[3] += IsotropicInterpolate(pIdx, d);

        d = (z_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[4] += IsotropicInterpolate(pIdx, d);

        d = (z_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[5] += IsotropicInterpolate(pIdx, d);
    }
}

inline void Evaluator::AnisotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius)
{
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    if (neighbors.empty())
    {
        return;
    }

    sample_radius = IS_CONST_RADIUS ? Radius : 
    GlobalRadius->at(*std::min_element(neighbors.begin(), neighbors.end(), [&](const int& a, const int& b) {
        return GlobalRadius->at(a) < GlobalRadius->at(b);
    }));

    Eigen::Vector3f x_up_pos(pos[0] + sample_radius, pos[1], pos[2]),
                    x_down_pos(pos[0] - sample_radius, pos[1], pos[2]),
                    y_up_pos(pos[0], pos[1] + sample_radius, pos[2]),
                    y_down_pos(pos[0], pos[1] - sample_radius, pos[2]),
                    z_up_pos(pos[0], pos[1], pos[2] + sample_radius),
                    z_down_pos(pos[0], pos[1], pos[2] - sample_radius);

    Eigen::Vector3f diff;
    for (int pIdx : neighbors)
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
