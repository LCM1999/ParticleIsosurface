#include <float.h>
#include "evaluator.h"
#include "surface_reconstructor.h"


Evaluator::Evaluator(SurfReconstructor* surf_constructor,
		std::vector<Eigen::Vector3f>* global_particles, 
        std::vector<float>* radiuses, float radius)
{
    constructor = surf_constructor;
	GlobalPoses = global_particles;
    NeighborFactor = constructor->getNeighborFactor();
    SmoothFactor = constructor->getSmoothFactor();
    GlobalRadius = radiuses;
    if (!IS_CONST_RADIUS)
    {
        GlobalRadius2.resize(GlobalRadius->size());
        GlobalRadius3.resize(GlobalRadius->size());
        GlobalInflunce2.resize(GlobalRadius->size());
        GlobalSigma.resize(GlobalRadius->size());
#pragma omp parallel for
        for (int i = 0; i < GlobalRadius->size(); i++)
        {
            float influnce = GlobalRadius->at(i) * SmoothFactor;
            GlobalRadius2[i] = pow(GlobalRadius->at(i), 2);
            GlobalRadius3[i] = GlobalRadius->at(i) * GlobalRadius2[i];
            GlobalInflunce2[i] = pow(influnce, 2);
            GlobalSigma[i] = (315 / (64 * pow(influnce, 9))) * inv_pi;
        }
    } else {
        Radius = radius;
        Radius2 = Radius * Radius;
        Radius3 = Radius * Radius2;
        Influnce2 = pow(Radius * SmoothFactor, 2);
        Sigma = (315 / (64 * pow(radius * SmoothFactor, 9))) * inv_pi;
    }

    if (CALC_P_NORMAL)
    {
        PariclesNormals.clear();
        PariclesNormals.resize(constructor->getGlobalParticlesNum(), Eigen::Vector3f(0, 0, 0)); 
    }
    
    GlobalSplash.clear();
    GlobalSplash.resize(constructor->getGlobalParticlesNum(), 0);
    GlobalxMeans.resize(constructor->getGlobalParticlesNum());
    if (USE_ANI)
    {
        GlobalGs.resize(constructor->getGlobalParticlesNum());
        GlobalDeterminant.resize(constructor->getGlobalParticlesNum());
    }
    
    compute_Gs_xMeans();
}

void Evaluator::SingleEval(const Eigen::Vector3f& pos, float& scalar)
{
	scalar = 0;
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    Eigen::Vector3f diff;
    for (int pIdx : neighbors)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }

        diff = pos - GlobalxMeans[pIdx];
        if (USE_ANI)
        {
            scalar += AnisotropicInterpolate(pIdx, diff);
        } else {
            scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
        }
    }
    scalar = constructor->getIsoValue() - ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

void Evaluator::SingleEvalWithGrad(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient, bool use_normalize, bool use_signed, bool grad_normalize)
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

    if (USE_ANI)
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
    float* sample_points, float* field_gradients, float cellsize, 
    bool& signchange, int oversample, bool grad_normalize)
{
    //assert(!sample_points.empty());
    bool origin_sign;
    float step = cellsize / oversample;
    for (int i = 0; i < pow(oversample+1, 3); i++)
    {
        Eigen::Vector4f p(sample_points[i*4 + 0], sample_points[i*4 + 1], sample_points[i*4 + 2], sample_points[i*4 + 3]);
        float scalar = 0.0f;
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
                if (USE_ANI)
                {
                    scalar += AnisotropicInterpolate(pIdx, p.head(3) - GlobalxMeans[pIdx]);
                } else {
                    scalar += IsotropicInterpolate(pIdx, (p.head(3) - GlobalxMeans[pIdx]).squaredNorm());
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
                    gradient[0] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (x == oversample)
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
                }

                next_idx = (z * (oversample + 1) * (oversample + 1) + (y + 1) * (oversample + 1) + x);
                last_idx = (z * (oversample + 1) * (oversample + 1) + (y - 1) * (oversample + 1) + x);
                if (y == 0)
                {
                    gradient[1] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (y == oversample)
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
                }

                next_idx = ((z + 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                last_idx = ((z - 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                if (z == 0)
                {
                    gradient[2] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (z == oversample)
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
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
    float k_value = 0;

    k_value += general_kernel(0.0, Influnce2, Sigma);
    k_value += general_kernel(4 * Radius2, Influnce2, Sigma) * 6;
    k_value += general_kernel(8 * Radius2, Influnce2, Sigma) * 12;
    k_value += general_kernel(12 * Radius2, Influnce2, Sigma) * 8;

    return Radius3 * k_value;
}

float Evaluator::CalculateMaxScalarVarR()
{
    float max_scalar = 0;
    auto searchers = constructor->getSearcher();
    unsigned int rId;
    // float r, radius2, influnce, influnce2, temp_scalar;
    float temp_scalar;
    for (const auto searcher : *(searchers->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        // r = searcher->Radius;
        // radius2 = r * r;
        // influnce = r * SmoothFactor;
        // influnce2 = influnce * influnce;
        temp_scalar += general_kernel(0.0, GlobalInflunce2[rId], GlobalSigma[rId]);
        temp_scalar += general_kernel(4 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 6;
        temp_scalar += general_kernel(8 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 12;
        temp_scalar += general_kernel(12 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 8;
        temp_scalar = GlobalRadius3[rId] * temp_scalar;
        if (temp_scalar > max_scalar)
        {
            max_scalar = temp_scalar;
        }
    }
    return max_scalar;
}

float Evaluator::RecommendIsoValueConstR(const float iso_factor)
{
    float k_value = 0.0;

    k_value += general_kernel(iso_factor * Radius2, Influnce2, Sigma);

    return (((Radius3 * k_value) 
    - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

float Evaluator::RecommendIsoValueVarR(const float iso_factor)
{
    float recommend = 0.0;
    auto searchers = constructor->getSearcher();
    unsigned int rId;
    // float r, radius2, influnce, influnce2, temp_scalar;
    float temp_scalar;

    for (const auto searcher : *(searchers->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        // r = searcher->Radius;
        // radius2 = r * r;
        // influnce = r * SmoothFactor;
        // influnce2 = influnce * influnce;
        temp_scalar += general_kernel(iso_factor * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]);
        temp_scalar = GlobalRadius3[rId] * temp_scalar;
        if (temp_scalar > recommend)
        {
            recommend = temp_scalar;
        }
    }
    return (recommend - constructor->getMinScalar()) / constructor->getMaxScalar() * 255;
}

void Evaluator::CalcParticlesNormal()
{
#pragma omp parallel for
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        float tempScalar = 0;
        Eigen::Vector3f tempGrad = Eigen::Vector3f::Zero();
        if (!CheckSplash(pIdx))
        {
            SingleEvalWithGrad(GlobalPoses->at(pIdx), tempScalar, tempGrad);
            PariclesNormals[pIdx][0] = tempGrad[0];
            PariclesNormals[pIdx][1] = tempGrad[1];
            PariclesNormals[pIdx][2] = tempGrad[2];
        }
    }
}

inline float Evaluator::general_kernel(float d2, float h2, float sigma)
{
    float p_dist = (d2 >= h2 ? 0.0 : pow(h2 - d2, 3));
    return p_dist * sigma;
}

inline float Evaluator::IsotropicInterpolate(const int pIdx, const float d2)
{
    float k_value = general_kernel(
        d2, 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));
	return (IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * k_value;
}

inline float Evaluator::AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff)
{
    float k_value = general_kernel(
        (GlobalGs[pIdx] * diff).squaredNorm(), 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));
    return (IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * (GlobalDeterminant[pIdx] * k_value);
}

inline void Evaluator::compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3f &xMean)
{
    float pR, pD2;
    pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
    // pD = SmoothFactor * pR;
    pD2 = IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx];
    float nR, nD, nD2, nI, nI2, d, d2, wj, wSum = 0;
    for (int nIdx : temp_neighbors)
    {
        if (nIdx == pIdx)
            continue;
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nD = SmoothFactor * nR * 1.4;
        nD2 = nD * nD;
        nI = nR * NeighborFactor;
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
        if (d2 <= std::max(pD2, nD2))
        {
            closer_neighbor++;
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
}

inline void Evaluator::compute_G(Eigen::Vector3f p, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G)
{
    const float invH = 1/SmoothFactor;
    float wSum = 0, d, wj;
    float nR, nI;

    Eigen::Vector3f wd = Eigen::Vector3f::Zero();
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    cov += Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
    wSum = 0.0f;
    for (int nIdx : neighbors)
    {
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nI = nR * NeighborFactor;
        d = (p - GlobalPoses->at(nIdx)).norm();
        wj = wij(d, nI);
        wSum += wj;
        wd = ((GlobalPoses->at(nIdx))) - xMean;
        cov += ((wd * wd.transpose()).cast<float>() * wj);
    }
    cov /= wSum;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Vector3f w = svd.singularValues();
    Eigen::Matrix3f v = svd.matrixV();
    
    w = Eigen::Vector3f(w.array().abs());
    const float maxSingularVal = w.maxCoeff();

    const float kr = 4.0;
    w[0] = std::max(w[0], maxSingularVal / kr);
    w[1] = std::max(w[1], maxSingularVal / kr);
    w[2] = std::max(w[2], maxSingularVal / kr);

    Eigen::Matrix3f invSigma = w.asDiagonal().inverse();
    // Compute G
    const float scale =
        std::pow(w[0] * w[1] * w[2], 1.0 / 3.0);  // volume preservation
    G = ((v * invSigma * u.transpose()) * invH * scale).cast<float>();
}

inline void Evaluator::compute_Gs_xMeans()
{
    const float invH = 1.0;
#pragma omp parallel for
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        std::vector<int> tempNeighbors;
        std::vector<int> neighbors;
        int closerNeigbors = 0;
        Eigen::Vector3f xMean = Eigen::Vector3f::Zero();
        Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
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

        compute_xMeans(pIdx, tempNeighbors, neighbors, closerNeigbors, xMean);

        if (closerNeigbors < 1)
        {
            G = Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
            GlobalSplash[pIdx] = 1;
        } 
        else
        {
            if (USE_ANI)
            {
                compute_G(GlobalPoses->at(pIdx), xMean, neighbors, G);
            }
        }
        
        GlobalxMeans[pIdx] = Eigen::Vector3f(xMean);
        if (USE_ANI)
        {
            GlobalGs[pIdx] = Eigen::Matrix3f(G);
            GlobalDeterminant[pIdx] = G.determinant();
        }
    }
}

inline float Evaluator::wij(float d, float r)
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
