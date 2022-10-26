
int tet_edge2vert[6][2] = 
{
	{0,2},
	{2,1},
	{0,1},
	{2,3},
	{0,3},
	{1,3}
};

int tet_trisNum[16] = {0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0};
/*
int tet_triStrips[16][5] = 
{
	{-1, -1, -1, -1, -1},
	{ 0,  2,  4, -1, -1},
	{ 2,  1,  5, -1, -1},
	{ 0,  1,  4,  5, -1},

	{ 1,  0,  3, -1, -1},
	{ 3,  1,  4,  2, -1},
	{ 0,  3,  2,  5, -1},
	{ 3,  5,  4, -1, -1},

	{ 3,  4,  5, -1, -1},
	{ 0,  2,  3,  5, -1},
	{ 3,  4,  1,  2, -1},
	{ 1,  3,  0, -1, -1},

	{ 0,  4,  1,  5, -1},
	{ 2,  5,  1, -1, -1},
	{ 0,  4,  2, -1, -1},
	{-1, -1, -1, -1, -1},
};
*/
int tet_triStrips[16][5] = 
{
	{0, -1, -1, -1, -1},
	{ 3, 0,  2,  4, -1},
	{ 3, 2,  1,  5, -1},
	{ 4, 0,  1,  4,  5},

	{ 3, 1,  0,  3, -1},
	{ 4, 3,  1,  4,  2},
	{ 4, 0,  3,  2,  5},
	{ 3, 3,  5,  4, -1},

	{ 3, 3,  4,  5, -1},
	{ 4, 0,  2,  3,  5},
	{ 4, 3,  4,  1,  2},
	{ 3, 1,  3,  0, -1},

	{ 4, 0,  4,  1,  5},
	{ 3, 2,  5,  1, -1},
	{ 3, 0,  4,  2, -1},
	{ 0, -1, -1, -1, -1},
};

int tet_tris[16][7] = 
{
	{-1, -1, -1, -1, -1, -1, -1},
	{ 0,  2,  4, -1, -1, -1, -1},
	{ 2,  1,  5, -1, -1, -1, -1},
	{ 0,  1,  4,  4,  1,  5, -1},

	{ 1,  0,  3, -1, -1, -1, -1},
	{ 3,  1,  4,  4,  1,  2, -1},
	{ 0,  3,  2,  2,  3,  5, -1},
	{ 3,  5,  4, -1, -1, -1, -1},

	{ 3,  4,  5, -1, -1, -1, -1},
	{ 0,  2,  3,  3,  2,  5, -1},
	{ 3,  4,  1,  1,  4,  2, -1},
	{ 1,  3,  0, -1, -1, -1, -1},

	{ 0,  4,  1,  1,  4,  5, -1},
	{ 2,  5,  1, -1, -1, -1, -1},
	{ 0,  4,  2, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1},
};
