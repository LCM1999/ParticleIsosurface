#include <iostream>
#include <fstream>
#include <stack>
#include "recorder.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "evaluator.h"

Recorder::Recorder(const std::string& output_dir, const std::string& frame_name, SurfReconstructor* surf_constructor)
{
    _Output_Dir = output_dir;
    _Frame_Name = frame_name;
	constructor = surf_constructor;
}

void Recorder::RecordProgress()
{
	//std::cout << record_name << std::endl;
	FILE* f = fopen((_Output_Dir + "/r_" + _Frame_Name + ".txt").c_str(), "w");
	if (constructor->getRoot() == nullptr)
	{
		return;
	}
	TNode* root = constructor->getRoot();

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
		//types += (std::to_string(temp_node->type) + " ");
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
	//fprintf(f, types.c_str());
	fprintf(f, "\"pos_x\",\"pos_y\",\"pos_z\",\"v\"\n");
	for (TNode* n : leaves_and_empty)
	{
		fprintf(f, "%f,%f,%f,%f\n", n->node[0], n->node[1], n->node[2], n->node[3]);
	}
	fclose(f);
}

void Recorder::RecordParticles()
{
	Evaluator* evaluator = constructor->getEvaluator();
	HashGrid* hashgrid = constructor->getHashGrid();
	Eigen::Vector3i xyz;
	FILE* f = fopen((_Output_Dir + "/rp_" + _Frame_Name + ".txt").c_str(), "w");
	fprintf(f, "\"x\",\"y\",\"z\",\"type\",\"hash\"\n");
	for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
	{
		hashgrid->CalcXYZIdx(evaluator->GlobalxMeans[pIdx], xyz);
		fprintf(f, "%f,%f,%f,%d,%d\n", evaluator->GlobalxMeans[pIdx][0], evaluator->GlobalxMeans[pIdx][1], evaluator->GlobalxMeans[pIdx][2], evaluator->CheckSplash(pIdx)?1:0, hashgrid->CalcCellHash(xyz));
	}
	fclose(f);
}

void Recorder::RecordFeatures()
{
	FILE* f = fopen((_Output_Dir + "/features_" + _Frame_Name + ".txt").c_str(), "w");
	if (constructor->getRoot() == nullptr)
	{
		return;
	}
	TNode* root = constructor->getRoot();
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
		//types += (std::to_string(temp_node->type) + " ");
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
	fprintf(f, "\"x\",\"y\",\"z\",\"scalar\", \"type\"\n");
	for (TNode* n : leaves_and_empty)
	{
		fprintf(f, "%f,%f,%f,%f,%d\n", n->node[0], n->node[1], n->node[2], n->node[3], (n->type == LEAF ? 1 : 0));
	}
	fclose(f);
}
