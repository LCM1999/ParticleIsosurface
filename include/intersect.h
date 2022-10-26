#pragma once
#include "vect.h"

bool intersect_aabb_tri(vect3f &mine, vect3f &maxe, vect3f &v0, vect3f &v1, vect3f &v2);
float dist2_aabb_pt(vect3f &mine, vect3f &maxe, vect3f &pt);
