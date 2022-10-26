#include "intersect.h"

bool intersect_aabb_tri(vect3f &mine, vect3f &maxe, vect3f &v0, vect3f &v1, vect3f &v2)
{
	//** use separating axes to determine intersection

	// create points
	vect3f *tri[3] = {&v0, &v1, &v2};
	vect3f *box_ext[2] = {&mine, &maxe};
	vect3f box[8];

	for (int i = 0; i < 8; i++)
		box[i]( (*box_ext[i&1])[0], (*box_ext[(i>>1)&1])[1], (*box_ext[i>>2])[2] );

	// create axes
	vect3f axes[13];
	axes[0](1,0,0);
	axes[1](0,1,0);
	axes[2](0,0,1);
	axes[3] = (v1 - v0) % (v2 - v0);
	for (int i = 0; i < 3; i++)
	{
		int ip = (i + 1) % 3;
		for (int j = 0; j < 3; j++)
		{
			axes[4 + i*3 + j] = axes[j] % ( (*tri[i]) - (*tri[ip]) );
		}
	}

	// check overlaps
	for (int i = 0; i < 13; i++)
	{
		// tri extents
		float tri_min, tri_max;
		tri_min = tri_max = (*tri[0]) * axes[i];
		for (int k = 1; k < 3; k++)
		{
			float d = (*tri[k]) * axes[i];
			if (d < tri_min)
				tri_min = d;
			if (d > tri_max)
				tri_max = d;
		}

		// box extents
		float box_min, box_max;
		box_min = box_max = box[0] * axes[i];
		for (int j = 1; j < 8; j++)
		{
			float d = box[j] * axes[i];
			if (d < box_min)
				box_min = d;
			if (d > box_max)
				box_max = d;
		}

		// if disjoint, they don't intersect
		if (box_max < tri_min || tri_max < box_min)
			return false;
	}

	return true;
}

float dist2_aabb_pt(vect3f &mine, vect3f &maxe, vect3f &pt)
{
	float d2 = 0;

	for (int i = 0; i < 3; i++)
	{
		float d = pt[i] - mine[i];
		if (d < 0)
		{
			d2 += d*d;
			continue;
		}

		d = pt[i] - maxe[i];
		if (d > 0)
			d2 += d*d;
	}

	return d2;
}
