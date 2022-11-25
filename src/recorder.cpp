#include <iostream>
#include <fstream>
#include <stack>
#include "recorder.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "evaluator.h"

Recorder::Recorder(std::string& output_dir, std::string& output_prefix, SurfReconstructor* surf_constructor)
{
    _Output_Dir = output_dir;
    _Output_Prefix = output_prefix;
	constructor = surf_constructor;
}

void Recorder::RecordProgress(const int& index)
{
	//std::cout << record_name << std::endl;
	FILE* f = fopen((_Output_Dir + "/r_" + _Output_Prefix + std::to_string(index) + ".txt").c_str(), "w");
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
	fprintf(f, "\"pos_x\",\"pos_y\",\"pos_z\",\"v\", \"id\"\n");
	for (TNode* n : leaves_and_empty)
	{
		fprintf(f, "%f,%f,%f,%f,%d\n", n->node[0], n->node[1], n->node[2], n->node[3], n->nId);
	}
	fclose(f);
}

void Recorder::RecordParticles(const int& index)
{
	Evaluator* evaluator = constructor->getEvaluator();
	int type = 0;
	float g_x, g_y, g_z;
	FILE* f = fopen((_Output_Dir + "/rp_" + _Output_Prefix + std::to_string(index) + ".txt").c_str(), "w");
	fprintf(f, "\"x\",\"y\",\"z\",\"type\",\"grad_x\",\"grad_y\",\"grad_z\"\n");
	for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
	{
		// if (evaluator->SurfaceNormals.find(pIdx) != evaluator->SurfaceNormals.end())
		// {
		// 	type = 2;
		// 	g_x = evaluator->SurfaceNormals[pIdx][0];
		// 	g_y = evaluator->SurfaceNormals[pIdx][1];
		// 	g_z = evaluator->SurfaceNormals[pIdx][2];
		// } else 
		if (evaluator->CheckSplash(pIdx))
		{
			type = 1;
			g_x = FLT_MAX;
			g_y = FLT_MAX;
			g_z = FLT_MAX;
		} else {
			type = 0;
			g_x = evaluator->PariclesNormals[pIdx][0];
			g_y = evaluator->PariclesNormals[pIdx][1];
			g_z = evaluator->PariclesNormals[pIdx][2];
		}
		fprintf(f, "%f,%f,%f,%d,%f,%f,%f\n", evaluator->GlobalPoses->at(pIdx)[0], evaluator->GlobalPoses->at(pIdx)[1], evaluator->GlobalPoses->at(pIdx)[2], type, g_x, g_y, g_z);
	}
	fclose(f);
}
