#include "timer.h"

#ifdef _WIN32

#include <windows.h>
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

#else

#include <sys/time.h>
#include <sys/resource.h>

double get_time()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

#endif
