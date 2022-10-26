#pragma once
#include "vect.h"
#include "iso_common.h"


template<class T> 
void function(vect<4,T> &p)
{
	vect<3,T> g;
	csg_root->eval(*(vect3f*)&p, p.v[3], g);
}

template <class T>
int sign(vect<4,T> &x)
{
	return x[3] > 0 ? 1 : -1;
}

template<class T>
vect<3, T> findZero(vect<4, T> &a, vect<4, T> &b)
{
	if (a[3] == 0)
		return a;
	else if (b[3] == 0)
		return b;
	else
	{
		T denom = 1.0f / (a[3] - b[3]);
		return vect<3, T>( (b[0]*a[3] - a[0]*b[3])*denom, (b[1]*a[3] - a[1]*b[3])*denom, (b[2]*a[3] - a[2]*b[3])*denom );
	}
}

template <class T>
void binarySearch(vect<4,T> &x, vect<4,T> &a, vect<4,T> &b, int depth = 0)
{

	if (depth >= FIND_ROOT_DEPTH)
	{
		x = findZero(a, b);
		//function(x);
		x[3] = 0;
		return;
	}
	else
	{
		x = (a + b) * .5;
		function(x);
	}

	vect<4,T> m = x;
	if (sign(a) != sign(m))
		binarySearch(x, a, m, depth+1);
	else
		binarySearch(x, m, b, depth+1);
}


template <class T>
vect<3,T> intersect_plane_line(vect<3,T> &plane_pos, vect<3,T> &plane_norm, vect<3,T> &line_pos, vect<3,T> &line_dir)
{
	T denom = plane_norm * line_dir;
	if (fabs(denom) < 1e-9)
		return vect<3,T>(1e30, 1e30, 1e30);

	T u = (plane_norm * (plane_pos - line_pos)) * (1.0 / denom);
	return line_pos + line_dir * u;
}

template <class T>
T intersect_line_line(vect<2,T> &p, vect<2,T> &d, vect<2,T> &q, vect<2,T> &e)
{
	// returns parameter value of second line
	return (d[1]*(-p[0] + q[0]) + d[0]*(p[1] - q[1]))/(-d[1]*e[0] + d[0]*e[1]);
}
