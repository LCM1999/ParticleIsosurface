#pragma once

#include "iso_common.h"
#include "global.h"
#include "array2d.h"
#include "vect.h"
#include "qefnorm.h"
#include "cube_arrays.h"
#include <float.h>
#include "rootfind.h"
#include "evaluator.h"

extern int tree_cells;

const short EMPTY = 0;
const short INTERNAL = 1;
const short LEAF = 2;
const short UNCERTAIN = 3;

const short CONTAIN = 0;
const short CONTAIN_IN = 1;
const short INTERSECT = 2;
const short DISJOINT = 3;

struct TNode
{
	TNode()
	{
		children[0] = children[1] = children[2] = children[3] = 
			children[4] = children[5] = children[6] = children[7] = 0;
		nId = tree_cells++;
		//tree_cells++;
		type = UNCERTAIN;
	}

	TNode(int id)
	{
		children[0] = children[1] = children[2] = children[3] =
			children[4] = children[5] = children[6] = children[7] = 0;
		nId = id;
		//tree_cells++;
		type = UNCERTAIN;
	}

	~TNode()
	{
		defoliate();
	}

	//vect4f verts[8];
	vect3f center;
	float half_length;
#ifdef USE_DMT
	vect4f edges[12];
	vect4f faces[6];
#endif // USE_DMT
	vect4f node;
	//bool empty;

	short depth = 0;
	unsigned __int64 nId;
	short type;

	TNode *children[8];

#ifdef USE_DMT
	bool changesSign(vect4f *verts, vect4f *edges, vect4f *faces, vect4f &node);
#endif // USE_DMT

#ifdef USE_DMC
	bool changeSignDMC(vect4f* verts);
#endif // USE_DMC

	//void eval(vect3f *grad, TNode *guide);
#ifdef USE_DMT
	bool changesSign(){return changesSign(verts, edges, faces, node);}
#endif // USE_DMT

#ifdef USE_DMC
	//bool changesSign() { return changeSignDMC(); };
#endif // USE_DMC

	int getWeight()
	{
		switch (type)
		{
		case EMPTY:
			return 0;
		case LEAF:
			return 1;
		case INTERNAL:
		{
			int sum = 0;
			for (TNode* child: children)
			{
				sum += child->getWeight();
			}
			return sum;
		}
		default:
			printf("ERROR: Get Uncertain Node During getWeight\n");
			exit(1);
		}
	}

	void defoliate();

	/*
	bool contains(vect3f &p)
	{
		for (int i = 0; i < 3; i++)
		{
			if (p[i] < verts[0][i] || p[i] > verts[7][i])
				return false;
		}
		return true;
	}
	bool is_outside()
	{
		if (verts[7][0] - verts[0][0] < 1.5)
		{
			for (int i = 0; i < 3; i++)
			{
				if (verts[0][i] < 0 || verts[7][i] > 1)
					return true;
			}
		}

		return false;
	}
	bool is_outside(){return is_outside((vect3f&)verts[0], (vect3f&)verts[7]);}

	bool is_outside(vect3f &mine, vect3f &maxe)
	{
		if (maxe[0] - mine[0] < 1.5)
		{
			for (int i = 0; i < 3; i++)
			{
				if (mine[i] < 0 || maxe[i] > 1)
					return true;
			}
		}

		return false;
	}
	*/

	bool is_leaf()
	{
		//assert(this != 0); 
		return type == LEAF || type == EMPTY;
		//return children[0] == 0;
		//if (children[0] == 0)
		//	return true;
		//return is_outside();
	}

	void prune(int depth = 0)
	{
		if (depth >= 2)
		{
			bool all_leaves = true;
			for (int i = 0; i < 8; i++)
			{
				if (children[i] && children[i]->children[0])
				{
					all_leaves = false;
					break;
				}
			}
			if (all_leaves)
				defoliate();
		}

		if (children[0])
			for (int i = 0; i < 8; i++)
				children[i]->prune(depth+1);
	}

	template <class T, class U>
	static double calcError(T& p, U& plane_norms, U& plane_pts)
	{
		assert(plane_norms.size() > 0);
		double err = 0;
		int plane_num = plane_norms.size();
		for (int i = 0; i < plane_norms.size(); i++)
		{
			if (((vect3f&)plane_norms[i]).isNan())
			{
				plane_num--;
				continue;
			}
			//double c = plane_norms[i] * p - plane_norms[i] * plane_pts[i];
			//err += c * c;
			err += squared(p[3] - (plane_norms[i] * (p - plane_pts[i]))) / (1 + plane_norms[i].length2());
		}
		return err;
	}

	template <class T>
	static auto squared(const T& t)
	{
		return t * t;
	}

	template <class T, class U, class V>
	static double calcErrorDMC(T& p, U& verts, V& verts_grad)
	{
		double err = 0;
		for (size_t i = 0; i < 8; i++)
		{
			err += squared(p[3] - verts[i][3] - verts_grad[i].dot((vect3f&)(p - verts[i])));
		}
		return err;
	}

#ifdef USE_DMT
	// NODES
	void vertNode(vect4f &p, vect3f *grad, float &qef_error, vect3f *ev = 0, int *ns_size = 0)
	{
		// debug
		vect3f mid_debug = 0;
		for (int i = 0; i < 8; i++)
		{
			vect4f &p4 = verts[i];
			mid_debug += p4;
		}
		mid_debug *= .125;
		bool do_debug = false;
		if (g.cursor == mid_debug)
			do_debug = true;

		// build QEF
		//float cellsize = verts[7][0] - verts[0][0];
		QEFNormal<double, 4> q;
		q.zero();

		vect4f mid = 0;

		vector<vect4f> plane_norms, plane_pts;

		for (int x = 0; x <= OVERSAMPLE_QEF; x++)
		{
			for (int y = 0; y <= OVERSAMPLE_QEF; y++)
			{
				for (int z = 0; z <= OVERSAMPLE_QEF; z++)
				{
					vect4f p;
					p[0] = (1 - float(x)/OVERSAMPLE_QEF)*verts[0][0] + (float(x)/OVERSAMPLE_QEF)*verts[7][0];
					p[1] = (1 - float(y)/OVERSAMPLE_QEF)*verts[0][1] + (float(y)/OVERSAMPLE_QEF)*verts[7][1];
					p[2] = (1 - float(z)/OVERSAMPLE_QEF)*verts[0][2] + (float(z)/OVERSAMPLE_QEF)*verts[7][2];

					vect5f pl;
					// csg_root->eval((vect3f&)p, p[3], (vect3f&)pl);
					evaluator->SingleEval((vect3f&)p, p[3], (vect3f&)pl);
					pl[3] = -1;
					pl[4] = -(p[0]*pl[0] + p[1]*pl[1] + p[2]*pl[2]) + p[3]; // -p*n

					q.combineSelf(vect5d(pl).v);

					mid += p;

					plane_pts.push_back(p);
					plane_norms.push_back(vect4f(pl[0], pl[1], pl[2], -1));
				}
			}
		}
		mid /= (OVERSAMPLE_QEF+1)*(OVERSAMPLE_QEF+1)*(OVERSAMPLE_QEF+1);

		//** calc minimizer
		if (do_debug)
			int alala=48932;

		// build system to solve
		const int n = 4;
		ArrayWrapper<double, n> A;
		double B [ n ];

		for (int i = 0; i < n; i++ )
		{
			int index = ( ( 2 * n + 3 - i ) * i ) / 2;
			for (int j = i; j < n; j++ )
			{
				A.data [ i ] [ j ] = q.data [ index + j - i ];
				A.data [ j ] [ i ] = A.data [ i ] [ j ];
			}

			B [ i ] = -q.data [ index + n - i ];
		}

		// minimize QEF constrained to cell
		const float border = BORDER * (verts[7][0]-verts[0][0]);
		bool is_out = true;
		double err = 1e30;
		vect3f mine(verts[0][0] + border, verts[0][1] + border, verts[0][2] + border);
		vect3f maxe(verts[7][0] - border, verts[7][1] - border, verts[7][2] - border);

		for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
		{
			if (cell_dim == 3)
			{
				// find minimal point
				vect4d rvalue;
				ArrayWrapper<double, n> inv;

				::matInverse<double, n> ( A, inv);

				for (int i = 0; i < n; i++ )
				{
					rvalue [ i ] = 0;
					for (int j = 0; j < n; j++ )
						rvalue [ i ] += inv.data [ j ] [ i ] * B [ j ];
				}

				p(rvalue[0], rvalue[1], rvalue[2], rvalue[3]);

				// check bounds
				if (p[0] >= mine[0] && p[0] <= maxe[0] &&
					p[1] >= mine[1] && p[1] <= maxe[1] &&
					p[2] >= mine[2] && p[2] <= maxe[2])
				{
					is_out = false;
					err = calcError(p, plane_norms, plane_pts);
				}
			}
			else if (cell_dim == 2)
			{
				for (int face = 0; face < 6; face++)
				{
					int dir = face / 2;
					int side = face % 2;
					vect3f corners[2] = {mine, maxe};

					// build constrained system
					ArrayWrapper<double, n+1> AC;
					double BC[n+1];
					for (int i = 0; i < n+1; i++)
					{
						for (int j = 0; j < n+1; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					AC.data[n][dir] = AC.data[dir][n] = 1;
					BC[n] = corners[side][dir];

					// find minimal point
					double rvalue[n+1];
					ArrayWrapper<double, n+1> inv;

					::matInverse<double, n+1> ( AC, inv);

					for (int i = 0; i < n+1; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+1; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect4f pc(rvalue[0], rvalue[1], rvalue[2], rvalue[3]);
				
					// check bounds
					int dp = (dir+1)%3;
					int dpp = (dir+2)%3;
					if (pc[dp] >= mine[dp] && pc[dp] <= maxe[dp] &&
						pc[dpp] >= mine[dpp] && pc[dpp] <= maxe[dpp])
					{
						is_out = false;
						double e = calcError(pc, plane_norms, plane_pts);
						if (e < err)
						{
							err = e;
							p = pc;
						}
					}
				}
			}
			else if (cell_dim == 1)
			{
				for (int edge = 0; edge < 12; edge++)
				{
					int dir = edge / 4;
					int side = edge % 4;
					vect3f corners[2] = {mine, maxe};

					// build constrained system
					ArrayWrapper<double, n+2> AC;
					double BC[n+2];
					for (int i = 0; i < n+2; i++)
					{
						for (int j = 0; j < n+2; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					int dp = (dir+1)%3;
					int dpp = (dir+2)%3;
					AC.data[n][dp] = AC.data[dp][n] = 1;
					AC.data[n+1][dpp] = AC.data[dpp][n+1] = 1;
					BC[n] = corners[side&1][dp];
					BC[n+1] = corners[side>>1][dpp];

					// find minimal point
					double rvalue[n+2];
					ArrayWrapper<double, n+2> inv;

					::matInverse<double, n+2> ( AC, inv);

					for (int i = 0; i < n+2; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+2; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect4f pc(rvalue[0], rvalue[1], rvalue[2], rvalue[3]);
				
					// check bounds
					if (pc[dir] >= mine[dir] && pc[dir] <= maxe[dir])
					{
						is_out = false;
						double e = calcError(pc, plane_norms, plane_pts);
						if (e < err)
						{
							err = e;
							p = pc;
						}
					}
				}
			}
			else if (cell_dim == 0)
			{
				for (int vertex = 0; vertex < 8; vertex++)
				{
					vect3f corners[2] = {mine, maxe};

					// build constrained system
					ArrayWrapper<double, n+3> AC;
					double BC[n+3];
					for (int i = 0; i < n+3; i++)
					{
						for (int j = 0; j < n+3; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					for (int i = 0; i < 3; i++)
					{
						AC.data[n+i][i] = AC.data[i][n+i] = 1;
						BC[n+i] = corners[(vertex>>i)&1][i];
					}
					// find minimal point
					double rvalue[n+3];
					ArrayWrapper<double, n+3> inv;

					::matInverse<double, n+3> ( AC, inv);

					for (int i = 0; i < n+3; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+3; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect4f pc(rvalue[0], rvalue[1], rvalue[2], rvalue[3]);
					// float v;
					// vect3f n;
					// mytree->eval((vect3f&)pc, pc[3], (vect3f)NULL);
					// check bounds
					double e = calcError(pc, plane_norms, plane_pts);
					if (e < err)
					{
						err = e;
						p = pc;
					}
				}
			}
		}
		if (p.v[0] - verts[0].v[0] < -TOLERANCE || p.v[0] - verts[7].v[0] > TOLERANCE ||
			p.v[1] - verts[0].v[1] < -TOLERANCE || p.v[1] - verts[7].v[1] > TOLERANCE ||
			p.v[2] - verts[0].v[2] < -TOLERANCE || p.v[2] - verts[7].v[2] > TOLERANCE)
		{
			printf("Out of node. \n");
			double guess[3];
			double minError = 10e8;
			for (int x = 0; x <= 10; x++)
			{
				for (int y = 0; y <= 10; y++)
				{
					for (int z = 0; z <= 10; z++)
					{
						vect3f sample_normal;
						vect4f sample;
						sample.v[0] = (1 - float(x) / 10) * verts[0][0] + (float(x) / 10) * verts[7][0];
						sample.v[1] = (1 - float(y) / 10) * verts[0][1] + (float(y) / 10) * verts[7][1];
						sample.v[2] = (1 - float(z) / 10) * verts[0][2] + (float(z) / 10) * verts[7][2];
						evaluator->SingleEval((vect3f&)sample, sample.v[3], sample_normal);

						double e = calcError(sample, plane_norms, plane_pts);
						if (e < minError)
						{
							minError = e;
							guess[0] = sample.v[0];
							guess[1] = sample.v[1];
							guess[2] = sample.v[2];
						}
					}
				}
			}
			//printf("%d \n", minIdx);
			p.v[0] = guess[0];
			p.v[1] = guess[1];
			p.v[2] = guess[2];
		}
		
		// eval
		//float qef_val = p[3];
		//function(p);
		evaluator->SingleEval(*(vect3f*)&p, p.v[3], grad[8]);
		//err = fabs(qef_val - p[3]);

		qef_error += err;
	}

	// FACES
	void vertFace(vect4f &p, vect3f *grad, float &qef_error, int which_face, vect3f *ev = 0)
	{
		//// debug
		//vect3f mid_debug = 0;
		//for (int i = 0; i < 4; i++)
		//{
		//	vect4f &p4 = verts[cube_face2vert[which_face][i]];
		//	mid_debug += p4;
		//}
		//mid_debug *= .25;
		//bool do_debug = false;
		//if (g.cursor == mid_debug)
		//	do_debug = true;

		// build QEF
		//float cellsize = verts[7][0] - verts[0][0];
		int xi = (cube_face2orient[which_face] + 1) % 3;
		int yi = (cube_face2orient[which_face] + 2) % 3;
		int zi = cube_face2orient[which_face];

		QEFNormal<double, 3> q;
		q.zero();

		vect3f mid = 0;

		vector<vect3f> plane_norms, plane_pts;

		vect4f &c0 = verts[cube_face2vert[which_face][0]], &c1 = verts[cube_face2vert[which_face][2]];
		for (int x = 0; x <= OVERSAMPLE_QEF; x++)
		{
			for (int y = 0; y <= OVERSAMPLE_QEF; y++)
			{
				vect4f p4;
				p4[xi] = (1 - float(x)/OVERSAMPLE_QEF)*c0[xi] + (float(x)/OVERSAMPLE_QEF)*c1[xi];
				p4[yi] = (1 - float(y)/OVERSAMPLE_QEF)*c0[yi] + (float(y)/OVERSAMPLE_QEF)*c1[yi];
				p4[zi] = c0[zi];

				vect3f n4;
				// csg_root->eval((vect3f&)p4, p4[3], (vect3f&)n4);
				evaluator->SingleEval((vect3f&)p4, p4[3], (vect3f&)n4);
				vect3f n3(n4[xi], n4[yi], -1);
				vect3f p3(p4[xi], p4[yi], p4[3]);
				vect4f pl = n3;
				pl[3] = -(p3*n3);	

				q.combineSelf(vect4d(pl).v);

				plane_pts.push_back(p3);
				plane_norms.push_back(n3);

				mid += p3;
			}
		}
		mid /= (OVERSAMPLE_QEF+1)*(OVERSAMPLE_QEF+1);

		// build system to solve
		const int n = 3;
		ArrayWrapper<double, n> A;
		double B [ n ];

		for (int i = 0; i < n; i++ )
		{
			int index = ( ( 2 * n + 3 - i ) * i ) / 2;
			for (int j = i; j < n; j++ )
			{
				A.data [ i ] [ j ] = q.data [ index + j - i ];
				A.data [ j ] [ i ] = A.data [ i ] [ j ];
			}

			B [ i ] = -q.data [ index + n - i ];
		}

		// minimize QEF constrained to cell
		const float border = BORDER * (verts[7][0]-verts[0][0]);
		bool is_out = true;
		double err = 1e30;
		vect2f mine(verts[cube_face2vert[which_face][0]][xi] + border, verts[cube_face2vert[which_face][0]][yi] + border);
		vect2f maxe(verts[cube_face2vert[which_face][2]][xi] - border, verts[cube_face2vert[which_face][2]][yi] - border);
		vect3f p3;

		for (int cell_dim = 2; cell_dim >= 0 && is_out; cell_dim--)
		{
			if (cell_dim == 2)
			{
				// find minimal point
				vect3d rvalue;
				ArrayWrapper<double, n> inv;

				::matInverse<double, n> ( A, inv);

				for (int i = 0; i < n; i++ )
				{
					rvalue [ i ] = 0;
					for (int j = 0; j < n; j++ )
						rvalue [ i ] += inv.data [ j ] [ i ] * B [ j ];
				}

				p3(rvalue[0], rvalue[1], rvalue[2]);

				// check bounds
				if (p3[0] >= mine[0] && p3[0] <= maxe[0] &&
					p3[1] >= mine[1] && p3[1] <= maxe[1])
				{
					is_out = false;
					err = calcError(p3, plane_norms, plane_pts);
				}
			}
			else if (cell_dim == 1)
			{
				for (int edge = 0; edge < 4; edge++)
				{
					int dir = edge / 2;
					int side = edge % 2;
					vect2f corners[2] = {mine, maxe};

					// build constrained system
					ArrayWrapper<double, n+1> AC;
					double BC[n+1];
					for (int i = 0; i < n+1; i++)
					{
						for (int j = 0; j < n+1; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					AC.data[n][dir] = AC.data[dir][n] = 1;
					BC[n] = corners[side][dir];

					// find minimal point
					double rvalue[n+1];
					ArrayWrapper<double, n+1> inv;

					::matInverse<double, n+1> ( AC, inv);

					for (int i = 0; i < n+1; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+1; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect4f pc(rvalue[0], rvalue[1], rvalue[2]);
				
					// check bounds
					int dp = (dir+1)%2;
					if (pc[dp] >= mine[dp] && pc[dp] <= maxe[dp])
					{
						is_out = false;
						double e = calcError(pc, plane_norms, plane_pts);
						if (e < err)
						{
							err = e;
							p3 = pc;
						}
					}
				}
			}
			else if (cell_dim == 0)
			{
				for (int vertex = 0; vertex < 4; vertex++)
				{
					vect2f corners[2] = {mine, maxe};

					// build constrained system
					ArrayWrapper<double, n+2> AC;
					double BC[n+2];
					for (int i = 0; i < n+2; i++)
					{
						for (int j = 0; j < n+2; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					for (int i = 0; i < 2; i++)
					{
						AC.data[n+i][i] = AC.data[i][n+i] = 1;
						BC[n+i] = corners[(vertex>>i)&1][i];
					}
					// find minimal point
					double rvalue[n+2];
					ArrayWrapper<double, n+2> inv;

					::matInverse<double, n+2> ( AC, inv);

					for (int i = 0; i < n+2; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+2; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect3f pc(rvalue[0], rvalue[1], rvalue[2]);
				
					// check bounds
					double e = calcError(pc, plane_norms, plane_pts);
					if (e < err)
					{
						err = e;
						p3 = pc;
					}
				}
			}
		}

		/*if (do_debug)
		{
			printf("ourSoln={%f,%f,%f}\n", p3[0], p3[1], p3[2]);

			printf("NMinimize[{\n");
			for (int i = 0; i < plane_norms.size(); i++)
			{
				if (i!=0)
					printf("+");
				printf("({%f,%f,%f}.{x,y,z}-\n{%f,%f,%f}.{%f,%f,%f})^2\n", 
					plane_norms[i][0], plane_norms[i][1], plane_norms[i][2],
					plane_norms[i][0], plane_norms[i][1], plane_norms[i][2],
					plane_pts[i][0], plane_pts[i][1], plane_pts[i][2]);
			}
			printf(",\n");
			printf("x>=%f&&x<=%f\n", mine[0], maxe[0]);
			printf("&&y>=%f&&y<=%f\n", mine[1], maxe[1]);
			printf("},{x,y,z}]\n");
		}*/

		// unproject back into 4 dimensions
		p = verts[cube_face2vert[which_face][0]];
		p[xi] = p3[0];
		p[yi] = p3[1];
		// function(p);
		evaluator->SingleEval(*(vect3f*)&p, p.v[3], grad[21 + which_face]);

		qef_error += err;

	}


	// EDGES
	void vertEdge(vect4f &p, vect3f *grad, float &qef_error, int which_edge)
	{
		//// debug
		//vect3f mid_debug = 0;
		//for (int i = 0; i < 2; i++)
		//{
		//	vect4f &p4 = verts[cube_edge2vert[which_edge][i]];
		//	mid_debug += p4;
		//}
		//mid_debug *= .5;
		//bool do_debug = false;
		//if (g.cursor == mid_debug)
		//	do_debug = true;

		// calc QEF
		//float cellsize = verts[7][0] - verts[0][0];
		int xi = cube_edge2orient[which_edge];
		int yi = (cube_edge2orient[which_edge] + 1) % 3;
		int zi = (cube_edge2orient[which_edge] + 2) % 3;

		QEFNormal<double, 2> q;
		q.zero();

		vect2f mid = 0;

		vector<vect2f> plane_norms, plane_pts;

		vect4f &c0 = verts[cube_edge2vert[which_edge][0]], &c1 = verts[cube_edge2vert[which_edge][1]];
		for (int i = 0; i <= OVERSAMPLE_QEF; i++)
		{
			vect4f p4;
			p4[xi] = (1 - float(i)/OVERSAMPLE_QEF)*c0[xi] + (float(i)/OVERSAMPLE_QEF)*c1[xi];
			p4[yi] = c0[yi];
			p4[zi] = c0[zi];

			vect3f g3;
			// csg_root->eval((vect3f&)p4, p4[3], (vect3f&)g3);
			evaluator->SingleEval((vect3f&)p4, p4[3], (vect3f&)g3);
			vect2f n2(g3[xi], -1);
			vect2f p2(p4[xi], p4[3]);
			vect3f pl = n2;
			pl[2] = -(p2*n2);

			q.combineSelf(vect3d(pl).v);

			plane_norms.push_back(n2);
			plane_pts.push_back(p2);

			mid += p2;
		}
		mid /= OVERSAMPLE_QEF+1;

		// build system to solve
		const int n = 2;
		ArrayWrapper<double, n> A;
		double B [ n ];

		for (int i = 0; i < n; i++ )
		{
			int index = ( ( 2 * n + 3 - i ) * i ) / 2;
			for (int j = i; j < n; j++ )
			{
				A.data [ i ] [ j ] = q.data [ index + j - i ];
				A.data [ j ] [ i ] = A.data [ i ] [ j ];
			}

			B [ i ] = -q.data [ index + n - i ];
		}

		// minimize QEF constrained to cell
		const float border = BORDER * (verts[7][0]-verts[0][0]);
		bool is_out = true;
		double err = 1e30;
		const float vmin = verts[cube_edge2vert[which_edge][0]][xi] + border;
		const float vmax = verts[cube_edge2vert[which_edge][1]][xi] - border;
		vect2f p2;

		for (int cell_dim = 1; cell_dim >= 0 && is_out; cell_dim--)
		{
			if (cell_dim == 1)
			{
				// find minimal point
				vect3d rvalue;
				ArrayWrapper<double, n> inv;

				::matInverse<double, n> ( A, inv);

				for (int i = 0; i < n; i++ )
				{
					rvalue [ i ] = 0;
					for (int j = 0; j < n; j++ )
						rvalue [ i ] += inv.data [ j ] [ i ] * B [ j ];
				}

				p2(rvalue[0], rvalue[1]);

				// check bounds
				if (p2[0] >= vmin && p2[0] <= vmax)
				{
					is_out = false;
					err = calcError(p2, plane_norms, plane_pts);
				}
			}
			else if (cell_dim == 0)
			{
				for (int vertex = 0; vertex < 2; vertex++)
				{
					float corners[2] = {vmin, vmax};

					// build constrained system
					ArrayWrapper<double, n+1> AC;
					double BC[n+1];
					for (int i = 0; i < n+1; i++)
					{
						for (int j = 0; j < n+1; j++)
						{
							AC.data[i][j] = (i < n && j < n ? A.data[i][j] : 0);
						}
						BC[i] = (i < n ? B[i] : 0);
					}

					for (int i = 0; i < 1; i++)
					{
						AC.data[n+i][i] = AC.data[i][n+i] = 1;
						BC[n+i] = corners[vertex>>i];
					}
					// find minimal point
					double rvalue[n+1];
					ArrayWrapper<double, n+1> inv;

					::matInverse<double, n+1> ( AC, inv);

					for (int i = 0; i < n+1; i++ )
					{
						rvalue [ i ] = 0;
						for (int j = 0; j < n+1; j++ )
							rvalue [ i ] += inv.data [ j ] [ i ] * BC [ j ];
					}

					vect2f pc(rvalue[0], rvalue[1]);
				
					// check bounds
					double e = calcError(pc, plane_norms, plane_pts);
					if (e < err)
					{
						err = e;
						p2 = pc;
					}
				}
			}
		}

		
		/*if (do_debug)
		{
			printf("ourSoln={%f,%f}\n", p2[0], p2[1]);

			printf("NMinimize[{\n");
			for (int i = 0; i < plane_norms.size(); i++)
			{
				if (i!=0)
					printf("+");
				printf("({%f,%f}.{x,y}-\n{%f,%f}.{%f,%f})^2\n", 
					plane_norms[i][0], plane_norms[i][1],
					plane_norms[i][0], plane_norms[i][1],
					plane_pts[i][0], plane_pts[i][1]);
			}
			printf(",\n");
			printf("x>=%f&&x<=%f\n", vmin, vmax);
			printf("},{x,y}]\n");
		}*/

		// unproject back into 4 dimensions
		p = verts[cube_edge2vert[which_edge][0]];
		p[xi] = p2[0];
		// function(p);
		evaluator->SingleEval(*(vect3f*)&p, p[3], grad[9 + which_edge]);

		qef_error += err;
		
	}
#endif // USE_DMT

	void vertAll(float& curv, bool& signchange, vect3f* grad, float& qef_error, bool pass_face, bool pass_edge)
	{
		bool origin_sign;
		signchange = false;

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};
		
		vect4f verts[8];
		for (Index i = 0; i < 8; i++)
		{
			verts[i][0] = center[0] + sign(i.x) * half_length;
			verts[i][1] = center[1] + sign(i.y) * half_length;
			verts[i][2] = center[2] + sign(i.z) * half_length;
		}

		//const float xcellsize = verts[7][0] - verts[0][0];
		//const float ycellsize = verts[7][1] - verts[0][1];
		//const float zcellsize = verts[7][2] - verts[0][2];
		//
		//const float border = BORDER * min(min(xcellsize, ycellsize), zcellsize);
		//
		//float x_sampling_step = P_RADIUS / 2, y_sampling_step = P_RADIUS / 2, z_sampling_step = P_RADIUS / 2;
		//
		//if (xcellsize > 2 * P_RADIUS)
		//{
		//	x_sampling_step = P_RADIUS;
		//}
		//		
		//if (ycellsize > 2 * P_RADIUS)
		//{
		//	y_sampling_step = P_RADIUS;
		//}
		//
		//if (zcellsize > 2 * P_RADIUS)
		//{
		//	z_sampling_step = P_RADIUS;
		//}
		//
		//const int xyz_oversample[3] = {
		//	int(ceil(xcellsize / x_sampling_step) + 1),
		//	int(ceil(ycellsize / y_sampling_step) + 1),
		//	int(ceil(zcellsize / z_sampling_step) + 1)
		//	//min(max(int(floor(xcellsize / x_sampling_step) + 1), 2), OVERSAMPLE_QEF),
		//	//min(max(int(floor(ycellsize / y_sampling_step) + 1), 2), OVERSAMPLE_QEF),
		//	//min(max(int(floor(zcellsize / z_sampling_step) + 1), 2), OVERSAMPLE_QEF)
		//};

		const float cellsize = 2 * half_length;

		const float border = BORDER * cellsize;

		float sampling_step = P_RADIUS / 2;

		int oversample = int(ceil(cellsize / sampling_step) + 1);

		if (depth < DEPTH_MIN)
		{
			//sampling_step = P_RADIUS;
			oversample = OVERSAMPLE_QEF;
		}

		bool is_out;
		double err;

		std::vector<vect3f> sample_points;
		std::vector<float> field_scalars;
		std::vector<vect3f> field_gradient;

		for (int z = 0; z <= oversample; z++)
		{
			for (int y = 0; y <= oversample; y++)
			{
				for (int x = 0; x <= oversample; x++)
				{
					sample_points.push_back(
						vect3f(
							(1 - float(x) / oversample) * verts[0][0] + (float(x) / oversample) * verts[7][0],
							(1 - float(y) / oversample) * verts[0][1] + (float(y) / oversample) * verts[7][1],
							(1 - float(z) / oversample) * verts[0][2] + (float(z) / oversample) * verts[7][2])
						);
					// evaluator->SingleEval(p, scalar, gradient);
					// sample_points.push_back(p);
					// field_scalars.push_back(scalar);
					// field_gradient.push_back(gradient);
					// 
					// if (!signchange)
					// {
					// 	if (z == 0  && y == 0 && z == 0)
					// 	{
					// 		origin_sign = scalar >= 0;
					// 	}
					// 	else
					// 	{
					// 		signchange = origin_sign ^ (scalar >= 0);
					// 	}
					// }	
				}
			}
		}

		evaluator->GridEval(sample_points, field_scalars, field_gradient, signchange, oversample);

		for (Index i = 0; i < 8; i++)
		{
			verts[i][3] = field_scalars[i.x * oversample + i.y * oversample * (oversample + 1) + i.z * oversample * (oversample + 1) * (oversample + 1)];
		}

		// calculate curvature
		vect3f norms(0, 0, 0);
		float area = 0;
		for (vect3f n : field_gradient)
		{
			norms += n;
			area += n.length();
		}

		curv = norms.length() / area;

		/*--------------------VERT NODE-----------------------*/
		QEFNormal<double, 4> node_q;
		node_q.zero();

		vect4f node_mid = 0;
		std::vector<vect3f> node_plane_norms, node_plane_pts;

		int node_index;
		for (int z = 0; z <= oversample; z++)
		{
			for (int y = 0; y <= oversample; y++)
			{
				for (int x = 0; x <= oversample; x++)
				{
					node_index = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
					vect3f p(sample_points[node_index][0], sample_points[node_index][1], sample_points[node_index][2]);
					vect5f pl;

					pl[0] = field_gradient[node_index][0];
					pl[1] = field_gradient[node_index][1];
					pl[2] = field_gradient[node_index][2];
					pl[3] = -1;
					pl[4] = -(p[0] * pl[0] + p[1] * pl[1] + p[2] * pl[2]) + field_scalars[node_index];

					node_q.combineSelf(vect5d(pl).v);

					node_mid += vect4f(sample_points[node_index][0], sample_points[node_index][1], sample_points[node_index][2], field_scalars[node_index]);

					node_plane_pts.push_back(p);
					node_plane_norms.push_back(vect3f(pl[0], pl[1], pl[2]));
				}
			}
		}
		node_mid /= (oversample + 1) * (oversample + 1) * (oversample + 1);

		// build system to solve
		const int node_n = 4;
		ArrayWrapper<double, node_n> node_A;
		double node_B[node_n];

		for (int i = 0; i < node_n; i++)
		{
			int index = ((2 * node_n + 3 - i) * i) / 2;
			for (int j = i; j < node_n; j++)
			{
				node_A.data[i][j] = node_q.data[index + j - i];
				node_A.data[j][i] = node_A.data[i][j];
			}
			node_B[i] = -node_q.data[index + node_n - i];
		}

		// minimize QEF constrained to cell

		is_out = true;
		err = 1e30;
		vect3f node_mine(verts[0][0] + border, verts[0][1] + border, verts[0][2] + border);
		vect3f node_maxe(verts[7][0] - border, verts[7][1] - border, verts[7][2] - border);

		vect4f pc(0, 0, 0, 0);
		vect3f pcg(0, 0, 0);

		for (int cell_dim = 3; cell_dim >= 0 && is_out; cell_dim--)
		{
			if (cell_dim == 3)
			{
				// find minimal point
				vect4d rvalue;
				ArrayWrapper<double, node_n> inv;

				::matInverse<double, node_n>(node_A, inv);

				for (int i = 0; i < node_n; i++)
				{
					rvalue[i] = 0;
					for (int j = 0; j < node_n; j++)
						rvalue[i] += inv.data[j][i] * node_B[j];
				}

				pc(rvalue[0], rvalue[1], rvalue[2]);
				evaluator->SingleEval((vect3f&)pc, pc[3], pcg);

				// check bounds
				if (pc[0] >= node_mine[0] && pc[0] <= node_maxe[0] &&
					pc[1] >= node_mine[1] && pc[1] <= node_maxe[1] &&
					pc[2] >= node_mine[2] && pc[2] <= node_maxe[2])
				{
					is_out = false;
#ifdef USE_DMC_ERR
					err = calcErrorDMC(pc, verts, grad);
#else
					err = calcError(pc, node_plane_norms, node_plane_pts);
#endif // USE_DMC_ERR
					node = pc;
				}
			}
			else if (cell_dim == 2)
			{
				for (int face = 0; face < 6; face++)
				{
					int dir = face / 2;
					int side = face % 2;
					vect3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					ArrayWrapper<double, node_n + 1> AC;
					double BC[node_n + 1];
					for (int i = 0; i < node_n + 1; i++)
					{
						for (int j = 0; j < node_n + 1; j++)
						{
							AC.data[i][j] = (i < node_n&& j < node_n ? node_A.data[i][j] : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					AC.data[node_n][dir] = AC.data[dir][node_n] = 1;
					BC[node_n] = corners[side][dir];

					// find minimal point
					double rvalue[node_n + 1];
					ArrayWrapper<double, node_n + 1> inv;

					::matInverse<double, node_n + 1>(AC, inv);

					for (int i = 0; i < node_n + 1; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 1; j++)
							rvalue[i] += inv.data[j][i] * BC[j];
					}

					pc(rvalue[0], rvalue[1], rvalue[2]);
					evaluator->SingleEval((vect3f&)pc, pc[3], pcg);

					// check bounds
					int dp = (dir + 1) % 3;
					int dpp = (dir + 2) % 3;
					if (pc[dp] >= node_mine[dp] && pc[dp] <= node_maxe[dp] &&
						pc[dpp] >= node_mine[dpp] && pc[dpp] <= node_maxe[dpp])
					{
						is_out = false;
#ifdef USE_DMC_ERR
						double e = calcErrorDMC(pc, verts, grad);
#else
						double e = calcError(pc, node_plane_norms, node_plane_pts);
#endif // USE_DMC_ERR
						if (e < err)
						{
							err = e;
							node = pc;
						}
					}
				}
			}
			else if (cell_dim == 1)
			{
				for (int edge = 0; edge < 12; edge++)
				{
					int dir = edge / 4;
					int side = edge % 4;
					vect3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					ArrayWrapper<double, node_n + 2> AC;
					double BC[node_n + 2];
					for (int i = 0; i < node_n + 2; i++)
					{
						for (int j = 0; j < node_n + 2; j++)
						{
							AC.data[i][j] = (i < node_n&& j < node_n ? node_A.data[i][j] : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					int dp = (dir + 1) % 3;
					int dpp = (dir + 2) % 3;
					AC.data[node_n][dp] = AC.data[dp][node_n] = 1;
					AC.data[node_n + 1][dpp] = AC.data[dpp][node_n + 1] = 1;
					BC[node_n] = corners[side & 1][dp];
					BC[node_n + 1] = corners[side >> 1][dpp];

					// find minimal point
					double rvalue[node_n + 2];
					ArrayWrapper<double, node_n + 2> inv;

					::matInverse<double, node_n + 2>(AC, inv);

					for (int i = 0; i < node_n + 2; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 2; j++)
							rvalue[i] += inv.data[j][i] * BC[j];
					}

					pc(rvalue[0], rvalue[1], rvalue[2]);
					evaluator->SingleEval((vect3f&)pc, pc[3], pcg);

					// check bounds
					if (pc[dir] >= node_mine[dir] && pc[dir] <= node_maxe[dir])
					{
						is_out = false;
#ifdef USE_DMC_ERR
						double e = calcErrorDMC(pc, verts, grad);
#else
						double e = calcError(pc, node_plane_norms, node_plane_pts);
#endif // USE_DMC_ERR
						if (e < err)
						{
							err = e;
							node = pc;
						}
					}
				}
			}
			else if (cell_dim == 0)
			{
				for (int vertex = 0; vertex < 8; vertex++)
				{
					vect3f corners[2] = { node_mine, node_maxe };

					// build constrained system
					ArrayWrapper<double, node_n + 3> AC;
					double BC[node_n + 3];
					for (int i = 0; i < node_n + 3; i++)
					{
						for (int j = 0; j < node_n + 3; j++)
						{
							AC.data[i][j] = (i < node_n&& j < node_n ? node_A.data[i][j] : 0);
						}
						BC[i] = (i < node_n ? node_B[i] : 0);
					}

					for (int i = 0; i < 3; i++)
					{
						AC.data[node_n + i][i] = AC.data[i][node_n + i] = 1;
						BC[node_n + i] = corners[(vertex >> i) & 1][i];
					}
					// find minimal point
					double rvalue[node_n + 3];
					ArrayWrapper<double, node_n + 3> inv;

					::matInverse<double, node_n + 3>(AC, inv);

					for (int i = 0; i < node_n + 3; i++)
					{
						rvalue[i] = 0;
						for (int j = 0; j < node_n + 3; j++)
							rvalue[i] += inv.data[j][i] * BC[j];
					}

					pc(rvalue[0], rvalue[1], rvalue[2]);
					evaluator->SingleEval((vect3f&)pc, pc[3], pcg);
					// check bounds
#ifdef USE_DMC_ERR
					double e = calcErrorDMC(pc, verts, grad);
#else
					double e = calcError(pc, node_plane_norms, node_plane_pts);
#endif // USE_DMC_ERR
					if (e < err)
					{
						err = e;
						node = pc;
					}
				}
			}
		}
		evaluator->SingleEval((vect3f&)node, node.v[3], grad[8]);
		if ((node.v[0] < verts[0].v[0] || node.v[0] > verts[7].v[0] ||
			node.v[1] < verts[0].v[1] || node.v[1] > verts[7].v[1] ||
			node.v[2] < verts[0].v[2] || node.v[2] > verts[7].v[2] 
			//|| (signchange && node.v[3] > (ISO_VALUE * 0.5))
			))
		{
			if (depth <= DEPTH_MIN)
			{
				node = (verts[0] + verts[7]) / 2;
				evaluator->SingleEval((vect3f&)node, node[3], grad[8]);
				//for (int i = 0; i < 6; i++)
				//{
				//	faces[i] = (verts[cube_face2vert[i][0]] + verts[cube_face2vert[i][2]]) / 2;
				//	evaluator->SingleEval((vect3f&)faces[i], faces[i][3], grad[21 + i]);
				//}
				//for (int i = 0; i < 12; i++)
				//{
				//	edges[i] = (verts[cube_edge2vert[i][0]] + verts[cube_edge2vert[i][1]]) / 2;
				//	evaluator->SingleEval((vect3f&)edges[i], edges[i][3], grad[9 + i]);
				//}
				return;
			}
			double minError = 1e30;
			int index;
			//int xyz_sample_num[3] = {
			//	max((int(xcellsize / (P_RADIUS / 2)) + 1), xyz_oversample[0]),
			//	max((int(ycellsize / (P_RADIUS / 2)) + 1), xyz_oversample[1]),
			//	max((int(zcellsize / (P_RADIUS / 2)) + 1), xyz_oversample[2])
			//};
			int sample_num = max((int(cellsize / (P_RADIUS / 2)) + 1), oversample);
			std::vector<vect3f> errors_samples;
			std::vector<float> errors_scalar;
			std::vector<float> errors_value;
			std::vector<vect3f> errors_grad;
			if (((sample_num == oversample)) || abs(node[3]) >= ISO_VALUE)
			{
				sample_num = oversample;
				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_samples = sample_points;
							errors_scalar = field_scalars;
							errors_value.push_back(
#ifdef USE_DMC_ERR
								calcErrorDMC(
									vect4f(
										errors_samples[index][0],
										errors_samples[index][1],
										errors_samples[index][2],
										errors_scalar[index]), 
									verts, grad)
#else
								calcError(
									vect4f(
										errors_samples[index][0], 
										errors_samples[index][1], 
										errors_samples[index][2], 
										errors_scalar[index]),
								node_plane_norms, node_plane_pts)
#endif // USE_DMC_ERR
							);
							errors_grad = field_gradient;
						}
					}
				}
			}
			else
			{
				printf("Out of node. ");
				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_samples.push_back(vect3f(
								(1 - float(x) / sample_num) * verts[0][0] + (float(x) / sample_num) * verts[7][0],
								(1 - float(y) / sample_num) * verts[0][1] + (float(y) / sample_num) * verts[7][1],
								(1 - float(z) / sample_num) * verts[0][2] + (float(z) / sample_num) * verts[7][2]));
						}
					}
				}
				std::vector<vect4f> errors_plane_norms, errors_plane_pts;
				//QEFNormal<double, 4> error_q;

				evaluator->GridEval(errors_samples, errors_scalar, errors_grad, signchange, sample_num);

				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							vect3f pl;
							vect3f p(errors_samples[index][0], errors_samples[index][1],
								errors_samples[index][2]);
							pl[0] = errors_grad[index][0];
							pl[1] = errors_grad[index][1];
							pl[2] = errors_grad[index][2];
							//pl[3] = -1;
							//pl[4] = -(
							//	errors_samples[index][0] * pl[0] + 
							//	errors_samples[index][1] * pl[1] + 
							//	errors_samples[index][2] * pl[2]) + errors_scalar[index];

							//error_q.combineSelf(vect5d(pl).v);

							errors_plane_pts.push_back(p);
							errors_plane_norms.push_back(pl);
						}
					}
				}

				for (int z = 0; z <= sample_num; z++)
				{
					for (int y = 0; y <= sample_num; y++)
					{
						for (int x = 0; x <= sample_num; x++)
						{
							index = z * (sample_num + 1) * (sample_num + 1) + y * (sample_num + 1) + x;
							errors_value.push_back(
#ifdef USE_DMC_ERR
								calcErrorDMC(
									vect4f(
										errors_samples[index][0],
										errors_samples[index][1],
										errors_samples[index][2],
										errors_scalar[index]),
									verts, grad)
#else
								calcError(
									vect4f(
										errors_samples[index][0],
										errors_samples[index][1],
										errors_samples[index][2],
										errors_scalar[index]),
									node_plane_norms, node_plane_pts)
#endif // USE_DMC_ERR
							);
						}
					}
				}

				norms(0, 0, 0);
				area = 0;
				for (vect3f n : errors_grad)
				{
					norms += n;
					area += n.length();
				}
				curv = norms.length() / area;
			}
			int minIndex = std::distance(errors_value.begin(),
				std::min_element(errors_value.begin(), errors_value.end()));
			node = errors_samples[minIndex];
			grad[8] = errors_grad[minIndex];
		}

		qef_error += err;
#ifdef USE_DMT
		if (!pass_face)
		{
			/*--------------------VERT FACE-----------------------*/
			std::vector<int> face_field_info;
			const int face_n = 3;
			for (int which_face = 0; which_face < 6; which_face++)
			{
				int xi = (cube_face2orient[which_face] + 1) % 3;
				int yi = (cube_face2orient[which_face] + 2) % 3;
				int zi = cube_face2orient[which_face];

				Index minv = cube_face2vert[which_face][0];
				Index maxv = cube_face2vert[which_face][2];

				int face_index;

				for (int z = minv.z * xyz_oversample[2]; z <= maxv.z * xyz_oversample[2]; z++)
				{
					for (int y = minv.y * xyz_oversample[1]; y <= maxv.y * xyz_oversample[1]; y++)
					{
						for (int x = minv.x * xyz_oversample[0]; x <= maxv.x * xyz_oversample[0]; x++)
						{
							face_index = z * (xyz_oversample[0] + 1) * (xyz_oversample[1] + 1) + y * (xyz_oversample[0] + 1) + x;
							face_field_info.push_back(face_index);
						}
					}
				}
				QEFNormal<double, 3> face_q;
				face_q.zero();
				vect3f face_mid = 0;
				std::vector<vect3f> face_plane_norms, face_plane_pts;

				for (int y = 0; y <= xyz_oversample[yi]; y++)
				{
					for (int x = 0; x <= xyz_oversample[xi]; x++)
					{
						face_index = (y * (xyz_oversample[xi] + 1) + x);
						vect4f p4(sample_points[face_field_info[face_index]][0],
							sample_points[face_field_info[face_index]][1],
							sample_points[face_field_info[face_index]][2],
							field_scalars[face_field_info[face_index]]);

						vect3f n4 = field_gradient[face_field_info[face_index]];
						vect3f n3(n4[xi], n4[yi], -1);
						vect3f p3(p4[xi], p4[yi], p4[3]);
						vect4f pl = n3;
						pl[3] = -(p3 * n3);

						face_q.combineSelf(vect4d(pl).v);

						face_plane_pts.push_back(p3);
						face_plane_norms.push_back(n3);

						face_mid += p3;
					}
				}
				face_mid /= (xyz_oversample[xi] + 1) * (xyz_oversample[yi] + 1);

				ArrayWrapper<double, face_n> A;
				double B[face_n];

				for (int i = 0; i < face_n; i++)
				{
					int index = ((2 * face_n + 3 - i) * i) / 2;
					for (int j = i; j < face_n; j++)
					{
						A.data[i][j] = face_q.data[index + j - i];
						A.data[j][i] = A.data[i][j];
					}

					B[i] = -face_q.data[index + face_n - i];
				}

				// minimize QEF constrained to cell
				is_out = true;
				err = 1e30;
				vect2f mine(verts[cube_face2vert[which_face][0]][xi] + border, verts[cube_face2vert[which_face][0]][yi] + border);
				vect2f maxe(verts[cube_face2vert[which_face][2]][xi] - border, verts[cube_face2vert[which_face][2]][yi] - border);
				vect3f p3;

				for (int cell_dim = 2; cell_dim >= 0 && is_out; cell_dim--)
				{
					if (cell_dim == 2)
					{
						// find minimal point
						vect3d rvalue;
						ArrayWrapper<double, face_n> inv;

						::matInverse<double, face_n>(A, inv);

						for (int i = 0; i < face_n; i++)
						{
							rvalue[i] = 0;
							for (int j = 0; j < face_n; j++)
								rvalue[i] += inv.data[j][i] * B[j];
						}

						p3(rvalue[0], rvalue[1], rvalue[2]);

						// check bounds
						if (p3[0] >= mine[0] && p3[0] <= maxe[0] &&
							p3[1] >= mine[1] && p3[1] <= maxe[1])
						{
							is_out = false;
							err = calcError(p3, face_plane_norms, face_plane_pts);
						}
					}
					else if (cell_dim == 1)
					{
						for (int edge = 0; edge < 4; edge++)
						{
							int dir = edge / 2;
							int side = edge % 2;
							vect2f corners[2] = { mine, maxe };

							// build constrained system
							ArrayWrapper<double, face_n + 1> AC;
							double BC[face_n + 1];
							for (int i = 0; i < face_n + 1; i++)
							{
								for (int j = 0; j < face_n + 1; j++)
								{
									AC.data[i][j] = (i < face_n&& j < face_n ? A.data[i][j] : 0);
								}
								BC[i] = (i < face_n ? B[i] : 0);
							}

							AC.data[face_n][dir] = AC.data[dir][face_n] = 1;
							BC[face_n] = corners[side][dir];

							// find minimal point
							double rvalue[face_n + 1];
							ArrayWrapper<double, face_n + 1> inv;

							::matInverse<double, face_n + 1>(AC, inv);

							for (int i = 0; i < face_n + 1; i++)
							{
								rvalue[i] = 0;
								for (int j = 0; j < face_n + 1; j++)
									rvalue[i] += inv.data[j][i] * BC[j];
							}

							vect4f pc(rvalue[0], rvalue[1], rvalue[2]);

							// check bounds
							int dp = (dir + 1) % 2;
							if (pc[dp] >= mine[dp] && pc[dp] <= maxe[dp])
							{
								is_out = false;
								double e = calcError(pc, face_plane_norms, face_plane_pts);
								if (e < err)
								{
									err = e;
									p3 = pc;
								}
							}
						}
					}
					else if (cell_dim == 0)
					{
						for (int vertex = 0; vertex < 4; vertex++)
						{
							vect2f corners[2] = { mine, maxe };

							// build constrained system
							ArrayWrapper<double, face_n + 2> AC;
							double BC[face_n + 2];
							for (int i = 0; i < face_n + 2; i++)
							{
								for (int j = 0; j < face_n + 2; j++)
								{
									AC.data[i][j] = (i < face_n&& j < face_n ? A.data[i][j] : 0);
								}
								BC[i] = (i < face_n ? B[i] : 0);
							}

							for (int i = 0; i < 2; i++)
							{
								AC.data[face_n + i][i] = AC.data[i][face_n + i] = 1;
								BC[face_n + i] = corners[(vertex >> i) & 1][i];
							}
							// find minimal point
							double rvalue[face_n + 2];
							ArrayWrapper<double, face_n + 2> inv;

							::matInverse<double, face_n + 2>(AC, inv);

							for (int i = 0; i < face_n + 2; i++)
							{
								rvalue[i] = 0;
								for (int j = 0; j < face_n + 2; j++)
									rvalue[i] += inv.data[j][i] * BC[j];
							}

							vect3f pc(rvalue[0], rvalue[1], rvalue[2]);

							// check bounds
							double e = calcError(pc, face_plane_norms, face_plane_pts);
							if (e < err)
							{
								err = e;
								p3 = pc;
							}
						}
					}
				}
				faces[which_face] = verts[cube_face2vert[which_face][0]];
				faces[which_face][xi] = p3[0];
				faces[which_face][yi] = p3[1];
				evaluator->SingleEval((vect3f&)faces[which_face], faces[which_face][3], grad[21 + which_face]);

				qef_error += err;
			}
		}
		else
		{
			for (int i = 0; i < 6; i++)
			{
				faces[i] = (verts[cube_face2vert[i][0]] + verts[cube_face2vert[i][2]]) / 2;
				evaluator->SingleEval((vect3f&)faces[i], faces[i][3], grad[21 + i]);
			}
		}
		if (!pass_edge)
		{
			/*--------------------VERT EDGE-----------------------*/
			std::vector<int> edge_field_info;
			const int edge_n = 2;
			for (int which_edge = 0; which_edge < 12; which_edge++)
			{
				int xi = cube_edge2orient[which_edge];
				int yi = (cube_edge2orient[which_edge] + 1) % 3;
				int zi = (cube_edge2orient[which_edge] + 2) % 3;

				Index minv = cube_edge2vert[which_edge][0];
				Index maxv = cube_edge2vert[which_edge][1];

				for (int z = minv.z * xyz_oversample[2]; z <= maxv.z * xyz_oversample[2]; z++)
				{
					for (int y = minv.y * xyz_oversample[1]; y <= maxv.y * xyz_oversample[1]; y++)
					{
						for (int x = minv.x * xyz_oversample[0]; x <= maxv.x * xyz_oversample[0]; x++)
						{
							int edge_index = z * (xyz_oversample[0] + 1) * (xyz_oversample[1] + 1) + y * (xyz_oversample[0] + 1) + x;
							edge_field_info.push_back(edge_index);
						}
					}
				}

				QEFNormal<double, 2> edge_q;
				edge_q.zero();
				vect2f edge_mid = 0;
				std::vector<vect2f> edge_plane_norms, edge_plane_pts;

				for (int i = 0; i <= xyz_oversample[xi]; i++)
				{
					vect4f p4(sample_points[edge_field_info[i]][0],
						sample_points[edge_field_info[i]][1],
						sample_points[edge_field_info[i]][2],
						field_scalars[edge_field_info[i]]);

					vect3f g3 = field_gradient[edge_field_info[i]];
					vect2f n2(g3[xi], -1);
					vect2f p2(p4[xi], p4[3]);
					vect3f pl = n2;
					pl[2] = -(p2 * n2);

					edge_q.combineSelf(vect3d(pl).v);

					edge_plane_norms.push_back(n2);
					edge_plane_pts.push_back(p2);

					edge_mid += p2;
				}
				edge_mid /= (xyz_oversample[xi] + 1);

				ArrayWrapper<double, edge_n> A;
				double B[edge_n];

				for (int i = 0; i < edge_n; i++)
				{
					int index = ((2 * edge_n + 3 - i) * i) / 2;
					for (int j = i; j < edge_n; j++)
					{
						A.data[i][j] = edge_q.data[index + j - i];
						A.data[j][i] = A.data[i][j];
					}

					B[i] = -edge_q.data[index + edge_n - i];
				}

				// minimize QEF constrained to cell
				is_out = true;
				err = 1e30;
				const float vmin = verts[cube_edge2vert[which_edge][0]][xi] + border;
				const float vmax = verts[cube_edge2vert[which_edge][1]][xi] - border;
				vect2f p2;

				for (int cell_dim = 1; cell_dim >= 0 && is_out; cell_dim--)
				{
					if (cell_dim == 1)
					{
						// find minimal point
						vect3d rvalue;
						ArrayWrapper<double, edge_n> inv;

						::matInverse<double, edge_n>(A, inv);

						for (int i = 0; i < edge_n; i++)
						{
							rvalue[i] = 0;
							for (int j = 0; j < edge_n; j++)
								rvalue[i] += inv.data[j][i] * B[j];
						}

						p2(rvalue[0], rvalue[1]);

						// check bounds
						if (p2[0] >= vmin && p2[0] <= vmax)
						{
							is_out = false;
							err = calcError(p2, edge_plane_norms, edge_plane_pts);
						}
					}
					else if (cell_dim == 0)
					{
						for (int vertex = 0; vertex < 2; vertex++)
						{
							float corners[2] = { vmin, vmax };

							// build constrained system
							ArrayWrapper<double, edge_n + 1> AC;
							double BC[edge_n + 1];
							for (int i = 0; i < edge_n + 1; i++)
							{
								for (int j = 0; j < edge_n + 1; j++)
								{
									AC.data[i][j] = (i < edge_n&& j < edge_n ? A.data[i][j] : 0);
								}
								BC[i] = (i < edge_n ? B[i] : 0);
							}

							for (int i = 0; i < 1; i++)
							{
								AC.data[edge_n + i][i] = AC.data[i][edge_n + i] = 1;
								BC[edge_n + i] = corners[vertex >> i];
							}
							// find minimal point
							double rvalue[edge_n + 1];
							ArrayWrapper<double, edge_n + 1> inv;

							::matInverse<double, edge_n + 1>(AC, inv);

							for (int i = 0; i < edge_n + 1; i++)
							{
								rvalue[i] = 0;
								for (int j = 0; j < edge_n + 1; j++)
									rvalue[i] += inv.data[j][i] * BC[j];
							}

							vect2f pc(rvalue[0], rvalue[1]);

							// check bounds
							double e = calcError(pc, edge_plane_norms, edge_plane_pts);
							if (e < err)
							{
								err = e;
								p2 = pc;
							}
						}
					}
				}
				edges[which_edge] = verts[cube_edge2vert[which_edge][0]];
				edges[which_edge][xi] = p2[0];
				evaluator->SingleEval((vect3f&)edges[which_edge], edges[which_edge][3], grad[9 + which_edge]);

				qef_error += err;
			}
		}
		else
		{
			for (int i = 0; i < 12; i++)
			{
				edges[i] = (verts[cube_edge2vert[i][0]] + verts[cube_edge2vert[i][1]]) / 2;
				evaluator->SingleEval((vect3f&)edges[i], edges[i][3], grad[9 + i]);
			}
		}
#endif // USE_DMT
	}

	int CountLeaves()
	{
		if (type == LEAF || type == EMPTY)
		{
			return 1;
		}
		int count = 0;
		for (TNode* child : children)
		{
			count += child->CountLeaves();
		}
		return count;
	}

};


struct Graph
{
	Graph() {};
	int vNum;
	int eNum;
	int ncon;
	map<unsigned __int64, int> vidx_map;
	vector<vector<int>> gAdj;
	//int* xadj;
	//int* adjncy;
	vector<int> vwgt;
	vector<int> ewgt;

	Graph(vector<TNode*>& layer_nodes, int vwn = 1)
	{
		vNum = 0;
		eNum = 0;
		ncon = vwn;
		//xadj = new int[vNum + 1];
		for (size_t i = 0; i < layer_nodes.size(); i++)
		{
			vidx_map[layer_nodes[i]->nId] = vNum;
			vNum++;
			for (size_t j = 0; j < ncon; j++)
			{
				vwgt.push_back(layer_nodes[i]->getWeight());
			}
		}
		gAdj.resize(vNum);
	}

	void appendEdge(const unsigned __int64 nId1, const unsigned __int64 nId2)
	{
		gAdj[vidx_map[nId1]].push_back(vidx_map[nId2]);
		eNum++;
		gAdj[vidx_map[nId2]].push_back(vidx_map[nId1]);
		eNum++;
	}

	void getXAdjAdjncy(int* xadj, int* adjncy)
	{
		int adjncyIdx = 0;
		for (size_t i = 0; i < vNum; i++)
		{
			xadj[i] = adjncyIdx;
			for (size_t j = 0; j < gAdj[i].size(); j++)
			{
				adjncy[xadj[i] + j] = gAdj[i][j];
			}
			adjncyIdx += gAdj[i].size();
		}
		xadj[vNum] = adjncyIdx;
		ewgt.resize(eNum, 1);
	}
};

const int edgevmap[12][2] = { {0,4},{1,5},{2,6},{3,7},{0,2},{1,3},{4,6},{5,7},{0,1},{2,3},{4,5},{6,7} };
const int edgemask[3] = { 5, 3, 6 };

// direction from parent st to each of the eight child st
// st is the corner of the cube with minimum (x,y,z) coordinates
//const int vertMap[8][3] = { {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1} };
const int vertMap[8][3] = { {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1} };

// map from the 6 faces of the cube to the 4 vertices that bound the face
const int faceMap[6][4] = { {4, 8, 5, 9}, {6, 10, 7, 11},{0, 8, 1, 10},{2, 9, 3, 11},{0, 4, 2, 6},{1, 5, 3, 7} };

// first used by cellProcCount()
// used in cellProcContour(). 
// between 8 child-nodes there are 12 faces.
// first two numbers are child-pairs, to be processed by faceProcContour()
// the last number is "dir" ?
const int cellProcFaceMask[12][3] = { {0,4,0},{1,5,0},{2,6,0},{3,7,0},{0,2,1},{4,6,1},{1,3,1},{5,7,1},{0,1,2},{2,3,2},{4,5,2},{6,7,2} };


// then used in cellProcContour() when calling edgeProc()
// between 8 children there are 6 common edges
// table lists the 4 children that share the edge
// the last number is "dir" ?
const int cellProcEdgeMask[6][5] = { {0,1,2,3,0},{4,5,6,7,0},{0,4,1,5,1},{2,6,3,7,1},{0,2,4,6,2},{1,3,5,7,2} };

// usde by faceProcCount()
const int faceProcFaceMask[3][4][3] = {
	{{4,0,0},{5,1,0},{6,2,0},{7,3,0}},
	{{2,0,1},{6,4,1},{3,1,1},{7,5,1}},
	{{1,0,2},{3,2,2},{5,4,2},{7,6,2}}
};
const int faceProcEdgeMask[3][4][6] = {
	{{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
	{{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
	{{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}}
};
const int edgeProcEdgeMask[3][2][5] = {
	{{3,2,1,0,0},{7,6,5,4,0}},
	{{5,1,4,0,1},{7,3,6,2,1}},
	{{6,4,2,0,2},{7,5,3,1,2}},
};
const int processEdgeMask[3][4] = { {3,2,1,0},{7,5,6,4},{11,10,9,8} };

const int dirCell[3][4][3] = {
	{{0,-1,-1},{0,-1,0},{0,0,-1},{0,0,0}},
	{{-1,0,-1},{-1,0,0},{0,0,-1},{0,0,0}},
	{{-1,-1,0},{-1,0,0},{0,-1,0},{0,0,0}}
};
const int dirEdge[3][4] = {
	{3,2,1,0},
	{7,6,5,4},
	{11,10,9,8}
};


void gen_iso_ours();
