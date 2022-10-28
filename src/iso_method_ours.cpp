#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stack>
#include <fstream>
#include <iterator>
#include <queue>

#include <omp.h>
#include <metis.h>
#include "iso_method_ours.h"
#include "index.h"
#include "timer.h"
#include "visitorextract.h"
#include "traverse.h"
#include "evaluator.h"
#include "utils.h"

int tree_cells = 0;
const std::string LOAD_RECORD_PATH = ".\\record_this.txt";
const std::string SAVE_RECORD_NAME = ".\\record_II.txt";

void record_progress(TNode* root_node, const char* record_name)
{
	FILE* f = fopen(record_name, "w");
	if (root_node == nullptr)
	{
		return;
	}
	TNode* root = root_node;
	fprintf(f, "%d\n", STATE);
	fprintf(f, "%d\n", tree_cells);

	std::stack<TNode*> node_stack;
	node_stack.push(root);
	std::vector<TNode*> leaves_and_empty;
	TNode* temp_node;
	std::string types = "", ids = "";
	while (!node_stack.empty())
	{
		temp_node = node_stack.top();
		node_stack.pop();
		if (temp_node == 0)
		{
			continue;
		}
		types += (std::to_string(temp_node->type) + " ");
		switch (temp_node->type)
		{
		case INTERNAL:
			for (int i = 7; i >= 0; i--)
			{
				node_stack.push(temp_node->children[i]);
			}
			break;
		case UNCERTAIN:
			break;
		default:
			leaves_and_empty.push_back(temp_node);
			break;
		}
	}
	fprintf(f, types.c_str());
	fprintf(f, "\n");
	for (TNode* n : leaves_and_empty)
	{
		fprintf(f, "%f %f %f %f\n", n->node[0], n->node[1], n->node[2], n->node[3]);
	}
	fclose(f);
}

void build_node(
	TNode* temp_node, std::vector<int>::iterator& node_it, std::vector<Eigen::Vector4f>::iterator& leaves_it)
{
	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	switch (*node_it)
	{
	case EMPTY:
	case LEAF:
		temp_node->type = *node_it;
		temp_node->node = *leaves_it;
		leaves_it++;
		return;
	case INTERNAL:
	{
		temp_node->type = INTERNAL;
		for (Index i; i < 8; i++)
		{
			temp_node->children[i] = new TNode(temp_node->nId * 8 + i + 1);
			temp_node->children[i]->depth = temp_node->depth + 1;
			temp_node->children[i]->half_length = temp_node->half_length / 2;
			temp_node->children[i]->center =
				temp_node->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * temp_node->half_length / 2);
			build_node(temp_node->children[i], ++node_it, leaves_it);
		}
		return;
	}
	case UNCERTAIN:
		temp_node->type = UNCERTAIN;
		return;
	default:
		break;
	}
}

void load_progress(TNode* loaded_tree, const char* record_path)
{
	std::ifstream ifn(record_path);

	std::string line;
	std::getline(ifn, line);
	STATE = std::stoi(line);

	std::getline(ifn, line);
	int temp_tree_cells = std::stoi(line);

	std::getline(ifn, line);
	std::vector<int> nodes;
	parseStringToElements<int>(&nodes, line, " ");

	std::getline(ifn, line);

	std::vector<Eigen::Vector4f> leaves;
	while (!line.empty())
	{
		std::vector<float> eles;
		parseStringToElements(&eles, line, " ");
		leaves.push_back(Eigen::Vector4f(eles[0], eles[1], eles[2], eles[3]));
		std::getline(ifn, line);
	}
	
	TNode* temp_node = loaded_tree;
	std::vector<int>::iterator node_it = nodes.begin();
	std::vector<Eigen::Vector4f>::iterator leaves_it = leaves.begin();

	build_node(temp_node, node_it, leaves_it);
	tree_cells = temp_tree_cells;
	ifn.close();
}

void eval(TNode* tnode, Eigen::Vector3f* grad, TNode* guide)
{
	if (tnode->nId % RECORD_STEP == 0 && STATE == 0)
	{
		if (NEED_RECORD)
		{
#pragma omp critical
			{
				record_progress(g.ourRoot, (CASE_PATH + "//" + RECORD_PREFIX + std::to_string(INDEX) + "_" + std::to_string(tnode->nId) + ".txt").c_str());
			}
		}
		printf("Record at %llu\n", tnode->nId);
	}
	float qef_error = 0, curv = 0;
	bool signchange = false, recur = false, next = false, empty;

	switch (tnode->type)
	{
	case EMPTY:
	case LEAF:
		return;
	case INTERNAL:
		next = true;
		break;
	case UNCERTAIN:
	{
		if (!guide || (guide && guide->children[0] == 0))
		{
			// evaluate QEF samples
			empty = true;
			Eigen::Vector3i min_xyz_idx, max_xyz_idx;
			hashgrid->CalcXYZIdx((tnode->center - Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length)), min_xyz_idx);
			hashgrid->CalcXYZIdx((tnode->center + Eigen::Vector3f(tnode->half_length, tnode->half_length, tnode->half_length)), max_xyz_idx);
			min_xyz_idx -= Eigen::Vector3i(1, 1, 1);
			max_xyz_idx += Eigen::Vector3i(1, 1, 1);
			__int64 temp_hash;
			for (int x = min_xyz_idx[0]; x <= max_xyz_idx[0] && empty; x++)
			{
				for (int y = min_xyz_idx[1]; y <= max_xyz_idx[1] && empty; y++)
				{
					for (int z = min_xyz_idx[2]; z <= max_xyz_idx[2] && empty; z++)
					{
						temp_hash = hashgrid->CalcCellHash(Eigen::Vector3i(x, y, z));
						if (temp_hash < 0)
							continue;
						if ((hashgrid->StartList.find(temp_hash) != hashgrid->StartList.end()) && (hashgrid->EndList.find(temp_hash) != hashgrid->EndList.end()))
						{
							if ((hashgrid->EndList[temp_hash] - hashgrid->StartList[temp_hash]) > 0)
							{
								empty = false;
							}
						}
					}
				}
			}
			if (empty)
			{
				tnode->node << tnode->center;
				evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
				tnode->type = EMPTY;
				//EMPTY_VOLUME += pow(2 * tnode->half_length, 3);
				//printStatus();
				return;
			}
			else if (tnode->depth < DEPTH_MIN)
			{
				tnode->node << tnode->center;
				evaluator->SingleEval((Eigen::Vector3f&)tnode->node, tnode->node[3], grad[8]);
			}
			else
			{
				tnode->vertAll(curv, signchange, grad, qef_error, true, true);
			}
		}

		// judge this node need calculate iso-surface
		float cellsize = 2 * tnode->half_length;

		if (!guide)
		{
			// check max/min sizes of cells

			// static float minsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MAX);
			bool issmall = (cellsize - (P_RADIUS)) < TOLERANCE;// || depth >= DEPTH_MAX;
			if (issmall)
			{
				// it's a leaf
				tnode->type = LEAF;
				//DONE_VOLUME += pow(2 * tnode->half_length, 3);
				//printStatus();
				return;
			}
			//static float maxsize = dynamic_cast<InternalNode*>(mytree->l)->lenn * pow(.5, DEPTH_MIN);
			bool isbig = tnode->depth < DEPTH_MIN;
			//
			//// check for a sign change
			//if (!isbig)
			//	signchange |= changeSignDMC();

			// check for qef error
			bool badqef = (qef_error / cellsize) > BAD_QEF;

			// check curvature
			bool badcurv = curv < FLATNESS;

			recur = isbig || (signchange && (badcurv || badqef));
		}
		else
		{
			recur = guide->children[0] != 0;
		}
	}
	break;
	default:
		break;
	}

	if (next)
	{
		if (guide)
		{
			for (Index i; i < 8; i++)
			{
				eval(tnode->children[i], grad, guide->children[i]);
			}
		}
		else
		{
			for (Index i; i < 8; i++)
			{
				eval(tnode->children[i], grad, 0);
			}
		}
	}
	else if (recur)
	{
		tnode->type = INTERNAL;
		// find points and function values in the subdivided cell
		Eigen::Vector3f g[3][3][3];
		float temp;
		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 3; y++)
			{
				for (int z = 0; z < 3; z++)
				{
					if (x == 1 || y == 1 || z == 1) {
						evaluator->SingleEval(tnode->center + (Eigen::Vector3f((x - 1), (y - 1), (z - 1)) * tnode->half_length), temp, g[x][y][z]);
					}
					else {
						g[x][y][z] = grad[Index(x >> 1, y >> 1, z >> 1)];
					}
				}
			}
		}

		auto sign = [&](unsigned int x)
		{
			return x ? 1 : -1;
		};

		// create children
		if (guide)
		{
#pragma omp parallel for
			for (int t = 0; t < 8; t++)
			{
				Index i = t;
				tnode->children[i]->depth = tnode->depth + 1;
				tnode->children[i]->half_length = tnode->half_length / 2;
				tnode->children[i]->center =
					tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
				for (Index j; j < 8; j++)
				{
					grad[j] = g[i.x + j.x][i.y + j.y][i.z + j.z];
				}
				eval(tnode->children[i], grad, guide->children[i]);
			}
		}
		else
		{
#pragma omp parallel for
			for (int t = 0; t < 8; t++)
			{
				Index i = t;
				tnode->children[i] = new TNode(tnode->nId * 8 + i + 1);
				tnode->children[i]->depth = tnode->depth + 1;
				tnode->children[i]->half_length = tnode->half_length / 2;
				tnode->children[i]->center =
					tnode->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * tnode->half_length / 2);
				for (Index j; j < 8; j++)
				{
					grad[j] = g[i.x + j.x][i.y + j.y][i.z + j.z];
				}
				eval(tnode->children[i], grad, 0);
			}
		}
	}
	else
	{
		tnode->type = LEAF;
		//DONE_VOLUME += pow(2 * tnode->half_length, 3);
		//printStatus();
	}
}

void get_division_depth(TNode* root, char& cdepth, std::vector<TNode*>& layer_nodes)
{
	std::queue<TNode*> layer_list;
	int layer_num = 0, layer_search_num = 0;
	int layer_depth = 0;
	TNode* temp;
	layer_list.push(root);

	auto queue2vect = [&](std::queue<TNode*>& q, std::vector<TNode*>& v)
	{
		while (!q.empty())
		{
			v.push_back(q.front());
			q.pop();
		}
	};

	if (OMP_THREADS_NUM <= 1)
	{
		cdepth = layer_depth;
		queue2vect(layer_list, layer_nodes);
		return;
	}

	do 
	{
		layer_depth++;
		layer_search_num = layer_list.size();
		for (size_t i = 0; i < layer_search_num; i++)
		{
			temp = layer_list.front();
			layer_list.pop();
			switch (temp->type)
			{
			case EMPTY:
			case LEAF:
				layer_list.push(temp);
				break;
			case INTERNAL:
				for (TNode* child: temp->children)
				{
					layer_list.push(child);
				}
				break;
			default:
				printf("Error: Get Uncertain Node During Octree division;");
				exit(1);
			}
		}
	} while (layer_list.size() < OMP_THREADS_NUM);	
	cdepth = layer_depth;
	queue2vect(layer_list, layer_nodes);
	return;
}

void gen_iso_ours()
{
	double t_start = get_time();

	TNode* guide;
	TNode* root;
	Mesh* m = &g.ourMesh;
	TNode* loaded_tree = nullptr;
	Eigen::Vector3f grad[9];

	if (LOAD_RECORD && g.ourRoot == nullptr)
	{
		loaded_tree = new TNode(0);
		loaded_tree->center = Eigen::Vector3f(RootCenter[0], RootCenter[1], RootCenter[2]);
		loaded_tree->half_length = RootHalfLength;
		load_progress(loaded_tree, LOAD_RECORD_PATH.c_str());
	}

	auto sign = [&](unsigned int x)
	{
		return x ? 1 : -1;
	};

	if (STATE == 0)
	{
		guide = g.ourRoot;
		printf("-= Calculating Tree Structure =-\n");
		if (LOAD_RECORD && g.ourRoot == nullptr)
		{
			root = loaded_tree;
		}
		else
		{
			root = new TNode(0);
			root->center = Eigen::Vector3f(RootCenter[0], RootCenter[1], RootCenter[2]);
			root->half_length = RootHalfLength;
			float temp;
			for (Index i; i < 8; i++) {
				evaluator->SingleEval(root->center + (Eigen::Vector3f(sign(i.x), sign(i.y), sign(i.z)) * root->half_length), temp, grad[i]);
			}
		}
		g.ourRoot = root;
	}
	else if (STATE == 1)
	{
		printf("-= Our Method =-\n");
		if (LOAD_RECORD && g.ourRoot == nullptr)
		{
			g.ourRoot = loaded_tree;
		}
		guide = g.ourRoot;
		root = guide;
	}
	else if (STATE == 2)
	{
		/*----------Make Graph----------*/
		char cdepth = 0;
		std::vector<TNode*> layer_nodes;
		get_division_depth(g.ourRoot, cdepth, layer_nodes);
		printf("cDepth = %d, layerNodesNum = %d\n", cdepth, layer_nodes.size());
		Graph layerGraph(layer_nodes);
		VisitorExtract gv(cdepth, &layerGraph);
		TraversalData gtd(g.ourRoot);
		traverse_node<trav_face>(gv, gtd);
		int* xadj = new int[layerGraph.vNum + 1];
		int* adjncy = new int[layerGraph.eNum * 2];
		layerGraph.getXAdjAdjncy(xadj, adjncy);

		std::vector<idx_t> part(layerGraph.vNum, 0);
		idx_t nParts = OMP_THREADS_NUM;
		idx_t obj_val;

		int ret = METIS_PartGraphRecursive(&layerGraph.vNum, &layerGraph.ncon, xadj, adjncy, layerGraph.vwgt.data(),
			NULL, layerGraph.ewgt.data(), &nParts, NULL, NULL, NULL, &obj_val, part.data());
		
		printf("-= Generate Surface =-\n");
		double t_gen_mesh = get_time();
		if (ret == rstatus_et::METIS_OK)
		{
			printf("-= METIS DIVISION SUCCESS =-\n");

		}
//		else
//		{
			printf("-= METIS ERROR, USE DEFAULT SINGLE THREAD =-\n");
			m->tris.reserve(1000000);
			VisitorExtract v(m);
			TraversalData td(g.ourRoot);
			traverse_node<trav_vert>(v, td);
		//}
		double t_alldone = get_time();
		printf("Time generating polygons = %f\n", t_alldone - t_gen_mesh);
		printf("Time total = %f\n", t_alldone - t_start);
		return;
	}

	eval(root, grad, guide);

	double t_finish = get_time();
	printf("Time generating tree = %f\n", t_finish - t_start);
	
	STATE++;
	if (NEED_RECORD)
	{
		record_progress(g.ourRoot, (CASE_PATH + "//" + RECORD_PREFIX + std::to_string(INDEX) + "_" + std::to_string(STATE) + ".txt").c_str());
	}
}

bool TNode::changeSignDMC(Eigen::Vector4f* verts)
{
	return  sign(verts[0]) != sign(verts[1]) ||
		sign(verts[0]) != sign(verts[2]) ||
		sign(verts[0]) != sign(verts[3]) ||
		sign(verts[0]) != sign(verts[4]) ||
		sign(verts[0]) != sign(verts[5]) ||
		sign(verts[0]) != sign(verts[6]) ||
		sign(verts[0]) != sign(verts[7]) ||
		sign(verts[0]) != sign(node);
}

void TNode::defoliate()
{
	for (int i = 0; i < 8; i++)
	{
		delete children[i];
		children[i] = 0;
	}
}

