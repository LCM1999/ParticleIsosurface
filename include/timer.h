#ifndef Timer_H
#define Timer_H
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
