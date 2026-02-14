#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <typeinfo>

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
//#include <Eigen/Dense>


static void printCudaMemUsage(const char* tag = "Current") {
    size_t free_mem = 0, total_mem = 0;
    // 1. 获取当前GPU的剩余/总显存（字节）
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    // 2. 单位转换：Byte -> MB (1MB=1024*1024Byte)，Byte->GB
    double free_mb = (double)free_mem / (1024 * 1024);
    double total_mb = (double)total_mem / (1024 * 1024);
    double used_mb = total_mb - free_mb;  // 已用显存=总-剩余

    double free_gb = free_mb / 1024;
    double total_gb = total_mb / 1024;
    double used_gb = used_mb / 1024;

    // 3. 格式化打印（保留2位小数，更整洁）
    std::cout << "=====================================" << std::endl;
    std::cout << tag << " GPU Memory Usage (GPU 0):" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total:  " << total_mb << " MB (" << total_gb << " GB)" << std::endl;
    std::cout << "Used:   " << used_mb << " MB (" << used_gb << " GB)" << std::endl;
    std::cout << "Free:   " << free_mb << " MB (" << free_gb << " GB)" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::resetiosflags(std::ios::fixed);  // 恢复输出格式
}

inline int sign(const float &x)
{
	return x > 0 ? 1 : -1;
}

void printMem();

void parseString(std::vector<std::string> *commList, const std::string& input, std::string sep);

void replaceAll(std::string &str, const std::string &olds, const std::string &news);

template <class T>
void parseStringToElements(std::vector<T>* elements, std::string& input, std::string sep)
{
	size_t pos = input.find(sep);

	while (pos != input.npos)
	{
		if (typeid(T) == typeid(int))
		{
			elements->push_back(stoi(input.substr(0, pos)));
		}
		else if (typeid(T) == typeid(float))
		{
			elements->push_back(stof(input.substr(0, pos)));
		}
		else
		{
			return;
		}
		input = input.substr(pos + 1);
		pos = input.find(sep);
	}

	if (!input.empty())
	{
		if (typeid(T) == typeid(int))
		{
			elements->push_back(stoi(input));
		}
		else if (typeid(T) == typeid(float))
		{
			elements->push_back(stof(input));
		}
		else
		{
			return;
		}
		input.clear();
	}
}
#endif

