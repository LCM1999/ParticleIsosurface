#pragma once
#ifndef EVALUATORGPU_CUH
#define EVALUATORGPU_CUH

#include "multi_level_searcher_gpu.cuh"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <coord_struct.h>

#include "evaluator.h"

struct EvaluatorGPU
{
    const float sqrt2 = 1.4142135623730950488016887242097;
    const float sqrt3 = 1.7320508075688772935274463415059;
    const float inv_pi = 0.31830988618379067153776752674503;

    float d_NEIGHBOR_FACTOR = 4.0;
    float d_SMOOTH_FACTOR = 2.0;
    float d_ISO_FACTOR = 1.9;
    float d_ISO_VALUE = 0.0;
    float d_MAX_SCALAR = -1.0;
    float d_MIN_SCALAR = 0.0;
    
    int d_GlobalParticlesNum = 0;
    float* d_GlobalRadiuses;
    float* d_GlobalInflunce2;
    float* d_GlobalSigma;
    char* d_GlobalSplash;
    cstoneOctree::Vec3f* d_PariclesNormals;
	cstoneOctree::Vec3f* d_GlobalxMeans;
    cstoneOctree::Mat3f* d_GlobalGs;
    float* d_GlobalDeterminant;

    // Constructor
    EvaluatorGPU(){}

    EvaluatorGPU(const Evaluator& evaluartor)
    {
        d_NEIGHBOR_FACTOR = evaluartor._NEIGHBOR_FACTOR;
        d_SMOOTH_FACTOR = evaluartor._SMOOTH_FACTOR;
        d_ISO_FACTOR = evaluartor._ISO_FACTOR;
        d_ISO_VALUE = evaluartor._ISO_VALUE;
        d_MAX_SCALAR = evaluartor._MAX_SCALAR;
        d_MIN_SCALAR = evaluartor._MIN_SCALAR;
        d_GlobalParticlesNum = evaluartor._GlobalParticlesNum;
        cudaMalloc((void**)&d_GlobalRadiuses, sizeof(float) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalInflunce2, sizeof(float) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalSigma, sizeof(float) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalSplash, sizeof(char) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_PariclesNormals, sizeof(cstoneOctree::Vec3f) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalxMeans, sizeof(cstoneOctree::Vec3f) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalGs, sizeof(cstoneOctree::Mat3f) * d_GlobalParticlesNum);
        cudaMalloc((void**)&d_GlobalDeterminant, sizeof(float) * d_GlobalParticlesNum);
        cudaMemcpy(d_GlobalRadiuses, evaluartor.GlobalRadius->data(), sizeof(float) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GlobalInflunce2, evaluartor.GlobalInflunce2.data(), sizeof(float) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GlobalSigma, evaluartor.GlobalSigma.data(), sizeof(float) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GlobalSplash, evaluartor.GlobalSplash.data(), sizeof(char) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        if (!evaluartor.PariclesNormals.empty())
        {
            cudaMemcpy(d_PariclesNormals, evaluartor.PariclesNormals.data(), sizeof(cstoneOctree::Vec3f) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_GlobalxMeans, evaluartor.GlobalxMeans.data(), sizeof(cstoneOctree::Vec3f) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GlobalGs, evaluartor.GlobalGs.data(), sizeof(cstoneOctree::Mat3f) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GlobalDeterminant, evaluartor.GlobalDeterminant.data(), sizeof(float) * d_GlobalParticlesNum, cudaMemcpyHostToDevice);
    }

    ~EvaluatorGPU() {
        // Free device memory
        cudaFree(d_GlobalInflunce2);
        cudaFree(d_GlobalSigma);
        cudaFree(d_GlobalSplash);
        cudaFree(d_PariclesNormals);
        cudaFree(d_GlobalxMeans);
        cudaFree(d_GlobalGs);
        cudaFree(d_GlobalDeterminant);
    }

    // Function to evaluate the GPU
    DEVICE float poly6_kernel(float d2, float h2, float sigma) {
        float p_dist = (d2 > h2 ? 0.0 : pow(h2 - d2, 3));
        return p_dist * sigma;
    };

    DEVICE cstoneOctree::Vec3f poly6_gradient_kernel(float d2, float h2, float sigma, const Vec3f diff) {
        Vec3f grad;
        grad[0] = sigma * (-6 * diff[0]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
        grad[1] = sigma * (-6 * diff[1]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
        grad[2] = sigma * (-6 * diff[2]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
        return grad;
    }

    DEVICE float AnisotropicInterpolate(const int pIdx, const cstoneOctree::Vec3f diff) {
        float k_value;
        k_value = poly6_kernel(
            (d_GlobalGs[pIdx] * diff).squaredNorm(), 
            (d_GlobalInflunce2[pIdx]), 
            (d_GlobalSigma[pIdx]));
        return (d_GlobalDeterminant[pIdx] * k_value);
    };

    DEVICE cstoneOctree::Vec3f AnisotropicInterpolateGrad(const int pIdx, const Vec3f diff) {
        Vec3f grad = poly6_gradient_kernel(
            (d_GlobalGs[pIdx] * diff).squaredNorm(), 
            (d_GlobalInflunce2[pIdx]), 
            (d_GlobalSigma[pIdx]), 
            (d_GlobalGs[pIdx] * diff));
        return (grad * d_GlobalDeterminant[pIdx]); 
    }

    DEVICE bool CheckSplash(const int& pIdx) {
        if (d_GlobalSplash[pIdx])
        {
            return true;
        }
        return false;
    };

    DEVICE float EvalInNodeCurv(const cstoneOctree::Vec3f box1, const cstoneOctree::Vec3f box2, 
        const int numNeighbors, const int* neighbors, float& minRadius, bool& empty) {
        cstoneOctree::Vec3f norms(0, 0, 0);
        float area = 0.0f;
        minRadius = FLT_MAX;
        empty = numNeighbors == 0;
        if (!empty)
        {
            bool allSplash = true;
            for (int i = 0; i < numNeighbors; i++)
            {
                int pIdx = neighbors[i];
                if (!CheckSplash(pIdx) 
                    && d_GlobalxMeans[pIdx].x > (box1.x - (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR)) 
                    && d_GlobalxMeans[pIdx].x < (box2.x + (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR)) 
                    && d_GlobalxMeans[pIdx].y > (box1.y - (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR)) 
                    && d_GlobalxMeans[pIdx].y < (box2.y + (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR)) 
                    && d_GlobalxMeans[pIdx].z > (box1.z - (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR)) 
                    && d_GlobalxMeans[pIdx].z < (box2.z + (d_GlobalRadiuses[pIdx] * d_SMOOTH_FACTOR))
                ) {
                    allSplash = false;
                    cstoneOctree::Vec3f tempNorm = d_PariclesNormals[pIdx];
					norms += tempNorm;
					area += tempNorm.norm();
                    minRadius = fminf(minRadius, d_GlobalRadiuses[pIdx]);
                }
            }
            empty = allSplash;
        }
        return (area == 0) ? 1.0 : (norms.norm() / area);
    }

    DEVICE void EvalInNode(const cstoneOctree::Vec3f box1, const cstoneOctree::Vec3f box2, 
        float* nodeSamplePoints, float* nodeSampleScalars, // node samples size is 3 * 3 * 3
        const int numNeighbors, const int* neighbors, bool& signChange) {
        bool originSign = true;
        for (int z = 0; z <= 2; z++)
        {
            for (int y = 0; y <= 2; y++)
            {
                for (int x = 0; x <= 2; x++)
                {
                    nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 0] = 
                    box1[0] * (1 - float(x) / 2) + box2[0] * (float(x) / 2);
                    nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 1] = 
                    box1[1] * (1 - float(y) / 2) + box2[1] * (float(y) / 2);
                    nodeSamplePoints[(z * (2+1) * (2+1) + y * (2+1) + x) * 3 + 2] = 
                    box1[2] * (1 - float(z) / 2) + box2[2] * (float(z) / 2);
                }
            }
        }

        for (int j = 0; j < 27; j++)
        {
            Vec3f diff;
            Vec3f samplePoint(nodeSamplePoints[j * 3 + 0], nodeSamplePoints[j * 3 + 1], nodeSamplePoints[j * 3 + 2]);
            nodeSampleScalars[j] = 0.0f;
            for (int k = 0; k < numNeighbors; k++)
            {
                int pIdx = neighbors[k];
                if (CheckSplash(pIdx))
                {
                    continue;
                }
                diff = samplePoint - d_GlobalxMeans[pIdx];
                nodeSampleScalars[j] += AnisotropicInterpolate(pIdx, diff);
            }

            nodeSampleScalars[j] = d_ISO_VALUE - nodeSampleScalars[j];
            originSign = (nodeSampleScalars[0] >= 0);
            if (!signChange)
            {
                signChange = originSign ^ (nodeSampleScalars[j] >= 0);
            }
        }
    }
};

#endif