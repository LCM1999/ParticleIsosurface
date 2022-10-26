#pragma once
#include "cube_arrays.h"

enum TraversalType
{
	trav_node,
	trav_face,
	trav_edge,
	trav_vert
};

// verts
template <TraversalType TT, class V, class T>
void traverse_vert(V &v, T &n000, T &n100, T &n010, T &n110, T &n001, T &n101, T &n011, T &n111) //nXYZ
{
	if (!v.on_vert(n000, n100, n010, n110, n001, n101, n011, n111))
		return;
	
	T c[8];
	n000.gen_trav(c[Index(0,0,0)], Index(1,1,1));
	n100.gen_trav(c[Index(1,0,0)], Index(0,1,1));
	n010.gen_trav(c[Index(0,1,0)], Index(1,0,1));
	n110.gen_trav(c[Index(1,1,0)], Index(0,0,1));
	n001.gen_trav(c[Index(0,0,1)], Index(1,1,0));
	n101.gen_trav(c[Index(1,0,1)], Index(0,1,0));
	n011.gen_trav(c[Index(0,1,1)], Index(1,0,0));
	n111.gen_trav(c[Index(1,1,1)], Index(0,0,0));

	traverse_vert<TT,V,T>(v, c[Index(0,0,0)],
							 c[Index(1,0,0)],
							 c[Index(0,1,0)],
							 c[Index(1,1,0)],
							 c[Index(0,0,1)],
							 c[Index(1,0,1)],
							 c[Index(0,1,1)],
							 c[Index(1,1,1)]);
}

// edges
template <TraversalType TT, class V, class T>
void traverse_edge_x(V &v, T &n00, T &n10, T &n01, T &n11) //nYZ
{
	if (!v.on_edge(n00, n10, n01, n11))
		return;

	T c[8];
	for (int i = 0; i < 2; i++)
	{
		n00.gen_trav(c[Index(i,0,0)], Index(i,1,1));
		n10.gen_trav(c[Index(i,1,0)], Index(i,0,1));
		n01.gen_trav(c[Index(i,0,1)], Index(i,1,0));
		n11.gen_trav(c[Index(i,1,1)], Index(i,0,0));
	}

	for (int i = 0; i < 2; i++)
	{
		traverse_edge_x<TT,V,T>(v, c[Index(i,0,0)], c[Index(i,1,0)], c[Index(i,0,1)], c[Index(i,1,1)]);
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}

template <TraversalType TT, class V, class T>
void traverse_edge_y(V &v, T &n00, T &n10, T &n01, T &n11) //nXZ
{
	if (!v.on_edge(n00, n10, n01, n11))
		return;
	
	T c[8];
	for (int i = 0; i < 2; i++)
	{
		n00.gen_trav(c[Index(0,i,0)], Index(1,i,1));
		n10.gen_trav(c[Index(1,i,0)], Index(0,i,1));
		n01.gen_trav(c[Index(0,i,1)], Index(1,i,0));
		n11.gen_trav(c[Index(1,i,1)], Index(0,i,0));
	}

	for (int i = 0; i < 2; i++)
	{
		traverse_edge_y<TT,V,T>(v, c[Index(0,i,0)], c[Index(1,i,0)], c[Index(0,i,1)], c[Index(1,i,1)]);
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}

template <TraversalType TT, class V, class T>
void traverse_edge_z(V &v, T &n00, T &n10, T &n01, T &n11) //nXY
{
	if (!v.on_edge(n00, n10, n01, n11))
		return;
	
	T c[8];
	for (int i = 0; i < 2; i++)
	{
		n00.gen_trav(c[Index(0,0,i)], Index(1,1,i));
		n10.gen_trav(c[Index(1,0,i)], Index(0,1,i));
		n01.gen_trav(c[Index(0,1,i)], Index(1,0,i));
		n11.gen_trav(c[Index(1,1,i)], Index(0,0,i));
	}

	for (int i = 0; i < 2; i++)
	{
		traverse_edge_z<TT,V,T>(v, c[Index(0,0,i)], c[Index(1,0,i)], c[Index(0,1,i)], c[Index(1,1,i)]);
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}

// faces
template <TraversalType TT, class V, class T>
void traverse_face_x(V &v, T &n0, T &n1)
{
	if (!v.on_face(n0, n1, 0))
		return;
	
	T c[8];
	for (Index i = 0; i < 4; i++)
	{
		n0.gen_trav(c[Index(0,i.x,i.y)], Index(1,i.x,i.y));
		n1.gen_trav(c[Index(1,i.x,i.y)], Index(0,i.x,i.y));
	}

	for (Index i; i < 4; i++)
		traverse_face_x<TT,V,T>(v, c[Index(0,i.x,i.y)], c[Index(1,i.x,i.y)]);
	
	if (TT >= trav_edge)
	{
		for (int i = 0; i < 2; i++)
		{
			traverse_edge_y<TT,V,T>(v, c[Index(0,i,0)], c[Index(1,i,0)], c[Index(0,i,1)], c[Index(1,i,1)]);
			traverse_edge_z<TT,V,T>(v, c[Index(0,0,i)], c[Index(1,0,i)], c[Index(0,1,i)], c[Index(1,1,i)]);
		}
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}

template <TraversalType TT, class V, class T>
void traverse_face_y(V &v, T &n0, T &n1)
{
	if (!v.on_face(n0, n1, 1))
		return;
	
	T c[8];
	for (Index i = 0; i < 4; i++)
	{
		n0.gen_trav(c[Index(i.x,0,i.y)], Index(i.x,1,i.y));
		n1.gen_trav(c[Index(i.x,1,i.y)], Index(i.x,0,i.y));
	}
	
	for (Index i; i < 4; i++)
		traverse_face_y<TT,V,T>(v, c[Index(i.x,0,i.y)], c[Index(i.x,1,i.y)]);
	
	if (TT >= trav_edge)
	{
		for (int i = 0; i < 2; i++)
		{
			traverse_edge_x<TT,V,T>(v, c[Index(i,0,0)], c[Index(i,1,0)], c[Index(i,0,1)], c[Index(i,1,1)]);
			traverse_edge_z<TT,V,T>(v, c[Index(0,0,i)], c[Index(1,0,i)], c[Index(0,1,i)], c[Index(1,1,i)]);
		}
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}

template <TraversalType TT, class V, class T>
void traverse_face_z(V &v, T &n0, T &n1)
{
	if (!v.on_face(n0, n1, 2))
		return;
	
	T c[8];
	for (Index i = 0; i < 4; i++)
	{
		n0.gen_trav(c[Index(i.x,i.y,0)], Index(i.x,i.y,1));
		n1.gen_trav(c[Index(i.x,i.y,1)], Index(i.x,i.y,0));
	}
	
	for (Index i; i < 4; i++)
		traverse_face_z<TT,V,T>(v, c[Index(i.x,i.y,0)], c[Index(i.x,i.y,1)]);
	
	if (TT >= trav_edge)
	{
		for (int i = 0; i < 2; i++)
		{
			traverse_edge_x<TT,V,T>(v, c[Index(i,0,0)], c[Index(i,1,0)], c[Index(i,0,1)], c[Index(i,1,1)]);
			traverse_edge_y<TT,V,T>(v, c[Index(0,i,0)], c[Index(1,i,0)], c[Index(0,i,1)], c[Index(1,i,1)]);
		}
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}


// nodes
template <TraversalType TT, class V, class T>
void traverse_node(V &v, T &td)
{
	if (!v.on_node(td))
		return;

	T c[8];
	for (Index i = 0; i < 8; i++)
	{
		td.gen_trav(c[i], i);
		traverse_node<TT,V,T>(v, c[i]);
	}

	if (TT >= trav_face)
	{
		for (Index i; i < 4; i++)
		{
			traverse_face_x<TT,V,T>(v, c[Index(0,i.x,i.y)], c[Index(1,i.x,i.y)]);
			traverse_face_y<TT,V,T>(v, c[Index(i.x,0,i.y)], c[Index(i.x,1,i.y)]);
			traverse_face_z<TT,V,T>(v, c[Index(i.x,i.y,0)], c[Index(i.x,i.y,1)]);
		}
	}
	
	if (TT >= trav_edge)
	{
		for (int i = 0; i < 2; i++)
		{
			traverse_edge_x<TT,V,T>(v, c[Index(i,0,0)], c[Index(i,1,0)], c[Index(i,0,1)], c[Index(i,1,1)]);
			traverse_edge_y<TT,V,T>(v, c[Index(0,i,0)], c[Index(1,i,0)], c[Index(0,i,1)], c[Index(1,i,1)]);
			traverse_edge_z<TT,V,T>(v, c[Index(0,0,i)], c[Index(1,0,i)], c[Index(0,1,i)], c[Index(1,1,i)]);
		}
	}

	if (TT >= trav_vert)
	{
		traverse_vert<TT,V,T>
			         (v, c[Index(0,0,0)],
						 c[Index(1,0,0)],
						 c[Index(0,1,0)],
						 c[Index(1,1,0)],
						 c[Index(0,0,1)],
						 c[Index(1,0,1)],
						 c[Index(0,1,1)],
						 c[Index(1,1,1)]);
	}
}
