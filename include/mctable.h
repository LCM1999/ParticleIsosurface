#pragma once

#include <vector>
using namespace std;

extern vector<vector<int> > mc_cycles[256];
extern int mc_edges[12][2];
void initMCTable();
