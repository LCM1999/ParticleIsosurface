#ifndef Timer_H
#define Timer_H

#ifdef _WIN32
#include <windows.h>
#include <Psapi.h>


struct timer
{
	LARGE_INTEGER start;
	timer() { QueryPerformanceCounter(&start); }
	double elapsed() const
	{
		LARGE_INTEGER end, freq;
		QueryPerformanceCounter(&end);
		QueryPerformanceFrequency(&freq);
		return (end.QuadPart - start.QuadPart) / double(freq.QuadPart);
	}
	void reset() { QueryPerformanceCounter(&start); }
};
#endif

#ifdef __linux__
#include <chrono> // C++11 标准库，用于更现代的计时
#include <iostream>
#include <ctime>   // clock_gettime
#include <iomanip> // std::fixed, std::setprecision

class timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

public:
  timer() : start(std::chrono::high_resolution_clock::now()) {}

  double elapsed() const {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

  void reset() { start = std::chrono::high_resolution_clock::now(); }
};

#endif

double get_time();

#endif
