#pragma once

#include "boundedarray.h"
#include "vect.h"

template <class T>
void clip_axis(BoundedArray<T> &p, BoundedArray<T> &a, BoundedArray<T> &b, char axis, float clip_val = 0)
{
	const int size = p.size();
	if (!size)
		return;

	T *vp = &p[size-1];
	float xp = vp->v[axis];
	bool inp = xp < clip_val;

	for (int i = 0; i < size; i++)
	{
		T &v = p[i];
		const float &x = v.v[axis];
		const bool in = x < clip_val;

		if (inp && in)
		{
			a.push_back(v);
		}
		else if (inp && !in)
		{
			const float div = x - xp;
			assert(div >= 0);
			if (div < 1e-6)
			{
				if (i != size-1)
				{
					a.push_back(v);
					b.push_back(v);
				}
			}
			else
			{
				T m = (v - (*vp))*(clip_val - xp)/div + *vp;

				a.push_back(m);

				b.push_back(m);
				b.push_back(v);
			}
		}
		else if (!inp && in)
		{
			const float div = x - xp;
			assert(div <= 0);
			if (div > -1e-6)
			{
				if (i != size-1)
				{
					a.push_back(v);
					b.push_back(v);
				}
			}
			else
			{
				T m = (v - (*vp))*(clip_val - xp)/div + *vp;

				a.push_back(m);
				a.push_back(v);
				
				b.push_back(m);
			}
		}
		else
		{
			b.push_back(v);
		}

		xp = x;
		vp = &v;
		inp = in;
	}
}


template <class T>
void clip_aabb(BoundedArray<T> &p, T &mine, T &maxe)
{
	BoundedArray<T> a, b;

	for (int i = 0; i < 3; i++)
	{
		a.clear();
		b.clear();
		clip_axis(p, a, b, i, mine[i]); // p -> b

		a.clear();
		p.clear();
		clip_axis(b, p, a, i, maxe[i]); // b -> p
	}
}
