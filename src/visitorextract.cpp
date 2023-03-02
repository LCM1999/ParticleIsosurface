#include <array>
#include "visitorextract.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "tet_arrays.h"
#include "iso_method_ours.h"
#include "evaluator.h"
#include "global.h"

template <class T, class U>
auto lerp(T x1, T x2, U ratio)
{
	return x1 + (x2 - x1) * ratio;
}

template <class T>
auto invlerp(T x1, T x2, T x)
{
	return (x - x1) / (x2 - x1);
}

TraversalData::TraversalData(TNode *t)
{
	n = t;
	depth = 0;
}
	
void TraversalData::gen_trav(TraversalData &c, Index i)
{
	if (!n->is_leaf())
	{
		c.n = n->children[i];
		c.depth = depth+1;
	}
	else
	{
		c = *this;
	}
}

VisitorExtract::VisitorExtract(SurfReconstructor* surf_constructor, Mesh* m_)
{
	constructor = surf_constructor;
	m = m_;
}

VisitorExtract::VisitorExtract(SurfReconstructor* surf_constructor, char cdepth, Graph* g_)
{
	constructor = surf_constructor;
	constrained_depth = cdepth;
	g = g_;
}

VisitorExtract::VisitorExtract(SurfReconstructor* surf_constructor, Mesh* m_, std::vector<TNode*>* part_)
{
	constructor = surf_constructor;
	m = m_;
	part = part_;
}

bool VisitorExtract::belong2part(TraversalData& td)
{
	for (TNode* n: *part)
	{
	}
	return false;
}

bool VisitorExtract::on_vert(TraversalData& a, TraversalData& b, TraversalData& c, TraversalData& d, TraversalData& aa, TraversalData& ba, TraversalData& ca, TraversalData& da)
{
	struct procedure
	{
		int trans[8];
		bool flip;
		int code;
	};
	static procedure table[256] =
	{
			/* 00000000 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00000000},
			/* 00000001 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00000001},
			/* 00000010 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00000001},
			/* 00000011 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00000011},
			/* 00000100 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00000001},
			/* 00000101 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b00000011},
			/* 00000110 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00000110},
			/* 00000111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00000111},
			/* 00001000 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00000001},
			/* 00001001 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00000110},
			/* 00001010 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b00000011},
			/* 00001011 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00000111},
			/* 00001100 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00000011},
			/* 00001101 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00000111},
			/* 00001110 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00000111},
			/* 00001111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00001111},
			/* 00010000 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00000001},
			/* 00010001 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b00000011},
			/* 00010010 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00000110},
			/* 00010011 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00000111},
			/* 00010100 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00000110},
			/* 00010101 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00000111},
			/* 00010110 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00010110},
			/* 00010111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00010111},
			/* 00011000 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00011000},
			/* 00011001 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00011001},
			/* 00011010 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b00011001},
			/* 00011011 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00011011},
			/* 00011100 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b00011001},
			/* 00011101 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b00011011},
			/* 00011110 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00011110},
			/* 00011111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00011111},
			/* 00100000 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00000001},
			/* 00100001 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b00000110},
			/* 00100010 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b00000011},
			/* 00100011 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b00000111},
			/* 00100100 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00011000},
			/* 00100101 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00011001},
			/* 00100110 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00011001},
			/* 00100111 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00011011},
			/* 00101000 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00000110},
			/* 00101001 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00010110},
			/* 00101010 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00000111},
			/* 00101011 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00010111},
			/* 00101100 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b00011001},
			/* 00101101 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00011110},
			/* 00101110 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b00011011},
			/* 00101111 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00011111},
			/* 00110000 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00000011},
			/* 00110001 */ {{4, 5, 0, 1, 6, 7, 2, 3}, false, 0b00000111},
			/* 00110010 */ {{5, 4, 1, 0, 7, 6, 3, 2}, true, 0b00000111},
			/* 00110011 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00001111},
			/* 00110100 */ {{4, 6, 0, 2, 5, 7, 1, 3}, true, 0b00011001},
			/* 00110101 */ {{0, 4, 1, 5, 2, 6, 3, 7}, false, 0b00011011},
			/* 00110110 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00011110},
			/* 00110111 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b00011111},
			/* 00111000 */ {{5, 7, 1, 3, 4, 6, 0, 2}, false, 0b00011001},
			/* 00111001 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b00011110},
			/* 00111010 */ {{1, 5, 0, 4, 3, 7, 2, 6}, true, 0b00011011},
			/* 00111011 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b00011111},
			/* 00111100 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00111100},
			/* 00111101 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00111101},
			/* 00111110 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b00111101},
			/* 00111111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b00111111},
			/* 01000000 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00000001},
			/* 01000001 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b00000110},
			/* 01000010 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00011000},
			/* 01000011 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00011001},
			/* 01000100 */ {{2, 6, 0, 4, 3, 7, 1, 5}, false, 0b00000011},
			/* 01000101 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b00000111},
			/* 01000110 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00011001},
			/* 01000111 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00011011},
			/* 01001000 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00000110},
			/* 01001001 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00010110},
			/* 01001010 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b00011001},
			/* 01001011 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00011110},
			/* 01001100 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00000111},
			/* 01001101 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00010111},
			/* 01001110 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00011011},
			/* 01001111 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00011111},
			/* 01010000 */ {{4, 6, 5, 7, 0, 2, 1, 3}, false, 0b00000011},
			/* 01010001 */ {{4, 6, 0, 2, 5, 7, 1, 3}, true, 0b00000111},
			/* 01010010 */ {{4, 5, 0, 1, 6, 7, 2, 3}, false, 0b00011001},
			/* 01010011 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b00011011},
			/* 01010100 */ {{6, 4, 2, 0, 7, 5, 3, 1}, false, 0b00000111},
			/* 01010101 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00001111},
			/* 01010110 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00011110},
			/* 01010111 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b00011111},
			/* 01011000 */ {{6, 7, 2, 3, 4, 5, 0, 1}, true, 0b00011001},
			/* 01011001 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b00011110},
			/* 01011010 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b00111100},
			/* 01011011 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b00111101},
			/* 01011100 */ {{2, 6, 0, 4, 3, 7, 1, 5}, false, 0b00011011},
			/* 01011101 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b00011111},
			/* 01011110 */ {{2, 0, 3, 1, 6, 4, 7, 5}, false, 0b00111101},
			/* 01011111 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b00111111},
			/* 01100000 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00000110},
			/* 01100001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00010110},
			/* 01100010 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00011001},
			/* 01100011 */ {{4, 5, 0, 1, 6, 7, 2, 3}, false, 0b00011110},
			/* 01100100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00011001},
			/* 01100101 */ {{4, 6, 0, 2, 5, 7, 1, 3}, true, 0b00011110},
			/* 01100110 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b00111100},
			/* 01100111 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b00111101},
			/* 01101000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00010110},
			/* 01101001 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b01101001},
			/* 01101010 */ {{7, 5, 3, 1, 6, 4, 2, 0}, true, 0b00011110},
			/* 01101011 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b01101011},
			/* 01101100 */ {{7, 6, 3, 2, 5, 4, 1, 0}, false, 0b00011110},
			/* 01101101 */ {{0, 2, 1, 3, 4, 6, 5, 7}, true, 0b01101011},
			/* 01101110 */ {{3, 7, 1, 5, 2, 6, 0, 4}, true, 0b00111101},
			/* 01101111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b01101111},
			/* 01110000 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00000111},
			/* 01110001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00010111},
			/* 01110010 */ {{4, 5, 0, 1, 6, 7, 2, 3}, false, 0b00011011},
			/* 01110011 */ {{4, 5, 0, 1, 6, 7, 2, 3}, false, 0b00011111},
			/* 01110100 */ {{4, 6, 0, 2, 5, 7, 1, 3}, true, 0b00011011},
			/* 01110101 */ {{4, 6, 0, 2, 5, 7, 1, 3}, true, 0b00011111},
			/* 01110110 */ {{4, 0, 6, 2, 5, 1, 7, 3}, false, 0b00111101},
			/* 01110111 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b00111111},
			/* 01111000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00011110},
			/* 01111001 */ {{0, 4, 2, 6, 1, 5, 3, 7}, true, 0b01101011},
			/* 01111010 */ {{5, 7, 4, 6, 1, 3, 0, 2}, true, 0b00111101},
			/* 01111011 */ {{0, 1, 4, 5, 2, 3, 6, 7}, true, 0b01101111},
			/* 01111100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00111101},
			/* 01111101 */ {{0, 2, 4, 6, 1, 3, 5, 7}, false, 0b01101111},
			/* 01111110 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b01111110},
			/* 01111111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b01111111},
			/* 10000000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00000001},
			/* 10000001 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00011000},
			/* 10000010 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b00000110},
			/* 10000011 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00011001},
			/* 10000100 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b00000110},
			/* 10000101 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00011001},
			/* 10000110 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00010110},
			/* 10000111 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00011110},
			/* 10001000 */ {{3, 7, 1, 5, 2, 6, 0, 4}, true, 0b00000011},
			/* 10001001 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00011001},
			/* 10001010 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b00000111},
			/* 10001011 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00011011},
			/* 10001100 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b00000111},
			/* 10001101 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00011011},
			/* 10001110 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00010111},
			/* 10001111 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00011111},
			/* 10010000 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00000110},
			/* 10010001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00011001},
			/* 10010010 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00010110},
			/* 10010011 */ {{5, 4, 1, 0, 7, 6, 3, 2}, true, 0b00011110},
			/* 10010100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00010110},
			/* 10010101 */ {{6, 4, 2, 0, 7, 5, 3, 1}, false, 0b00011110},
			/* 10010110 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b01101001},
			/* 10010111 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b01101011},
			/* 10011000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00011001},
			/* 10011001 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b00111100},
			/* 10011010 */ {{5, 7, 1, 3, 4, 6, 0, 2}, false, 0b00011110},
			/* 10011011 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b00111101},
			/* 10011100 */ {{6, 7, 2, 3, 4, 5, 0, 1}, true, 0b00011110},
			/* 10011101 */ {{2, 6, 0, 4, 3, 7, 1, 5}, false, 0b00111101},
			/* 10011110 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b01101011},
			/* 10011111 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b01101111},
			/* 10100000 */ {{5, 7, 4, 6, 1, 3, 0, 2}, true, 0b00000011},
			/* 10100001 */ {{5, 4, 1, 0, 7, 6, 3, 2}, true, 0b00011001},
			/* 10100010 */ {{5, 7, 1, 3, 4, 6, 0, 2}, false, 0b00000111},
			/* 10100011 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b00011011},
			/* 10100100 */ {{7, 6, 3, 2, 5, 4, 1, 0}, false, 0b00011001},
			/* 10100101 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b00111100},
			/* 10100110 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b00011110},
			/* 10100111 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b00111101},
			/* 10101000 */ {{7, 5, 3, 1, 6, 4, 2, 0}, true, 0b00000111},
			/* 10101001 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00011110},
			/* 10101010 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00001111},
			/* 10101011 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b00011111},
			/* 10101100 */ {{3, 7, 1, 5, 2, 6, 0, 4}, true, 0b00011011},
			/* 10101101 */ {{3, 1, 2, 0, 7, 5, 6, 4}, true, 0b00111101},
			/* 10101110 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b00011111},
			/* 10101111 */ {{1, 3, 0, 2, 5, 7, 4, 6}, false, 0b00111111},
			/* 10110000 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00000111},
			/* 10110001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00011011},
			/* 10110010 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00010111},
			/* 10110011 */ {{5, 4, 1, 0, 7, 6, 3, 2}, true, 0b00011111},
			/* 10110100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00011110},
			/* 10110101 */ {{4, 6, 5, 7, 0, 2, 1, 3}, false, 0b00111101},
			/* 10110110 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b01101011},
			/* 10110111 */ {{1, 0, 5, 4, 3, 2, 7, 6}, false, 0b01101111},
			/* 10111000 */ {{5, 7, 1, 3, 4, 6, 0, 2}, false, 0b00011011},
			/* 10111001 */ {{5, 1, 7, 3, 4, 0, 6, 2}, true, 0b00111101},
			/* 10111010 */ {{5, 7, 1, 3, 4, 6, 0, 2}, false, 0b00011111},
			/* 10111011 */ {{1, 5, 3, 7, 0, 4, 2, 6}, false, 0b00111111},
			/* 10111100 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00111101},
			/* 10111101 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b01111110},
			/* 10111110 */ {{1, 3, 5, 7, 0, 2, 4, 6}, true, 0b01101111},
			/* 10111111 */ {{1, 0, 3, 2, 5, 4, 7, 6}, true, 0b01111111},
			/* 11000000 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00000011},
			/* 11000001 */ {{6, 4, 2, 0, 7, 5, 3, 1}, false, 0b00011001},
			/* 11000010 */ {{7, 5, 3, 1, 6, 4, 2, 0}, true, 0b00011001},
			/* 11000011 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00111100},
			/* 11000100 */ {{6, 7, 2, 3, 4, 5, 0, 1}, true, 0b00000111},
			/* 11000101 */ {{2, 6, 3, 7, 0, 4, 1, 5}, true, 0b00011011},
			/* 11000110 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b00011110},
			/* 11000111 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00111101},
			/* 11001000 */ {{7, 6, 3, 2, 5, 4, 1, 0}, false, 0b00000111},
			/* 11001001 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00011110},
			/* 11001010 */ {{3, 7, 2, 6, 1, 5, 0, 4}, false, 0b00011011},
			/* 11001011 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b00111101},
			/* 11001100 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00001111},
			/* 11001101 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b00011111},
			/* 11001110 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b00011111},
			/* 11001111 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b00111111},
			/* 11010000 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00000111},
			/* 11010001 */ {{4, 6, 5, 7, 0, 2, 1, 3}, false, 0b00011011},
			/* 11010010 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00011110},
			/* 11010011 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00111101},
			/* 11010100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00010111},
			/* 11010101 */ {{6, 4, 2, 0, 7, 5, 3, 1}, false, 0b00011111},
			/* 11010110 */ {{2, 6, 0, 4, 3, 7, 1, 5}, false, 0b01101011},
			/* 11010111 */ {{2, 0, 6, 4, 3, 1, 7, 5}, true, 0b01101111},
			/* 11011000 */ {{6, 7, 2, 3, 4, 5, 0, 1}, true, 0b00011011},
			/* 11011001 */ {{6, 2, 4, 0, 7, 3, 5, 1}, true, 0b00111101},
			/* 11011010 */ {{7, 5, 6, 4, 3, 1, 2, 0}, false, 0b00111101},
			/* 11011011 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b01111110},
			/* 11011100 */ {{6, 7, 2, 3, 4, 5, 0, 1}, true, 0b00011111},
			/* 11011101 */ {{2, 6, 0, 4, 3, 7, 1, 5}, false, 0b00111111},
			/* 11011110 */ {{2, 3, 6, 7, 0, 1, 4, 5}, false, 0b01101111},
			/* 11011111 */ {{2, 3, 0, 1, 6, 7, 4, 5}, true, 0b01111111},
			/* 11100000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00000111},
			/* 11100001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00011110},
			/* 11100010 */ {{5, 7, 4, 6, 1, 3, 0, 2}, true, 0b00011011},
			/* 11100011 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00111101},
			/* 11100100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00011011},
			/* 11100101 */ {{6, 4, 7, 5, 2, 0, 3, 1}, true, 0b00111101},
			/* 11100110 */ {{7, 3, 5, 1, 6, 2, 4, 0}, false, 0b00111101},
			/* 11100111 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b01111110},
			/* 11101000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00010111},
			/* 11101001 */ {{3, 7, 1, 5, 2, 6, 0, 4}, true, 0b01101011},
			/* 11101010 */ {{7, 5, 3, 1, 6, 4, 2, 0}, true, 0b00011111},
			/* 11101011 */ {{3, 1, 7, 5, 2, 0, 6, 4}, false, 0b01101111},
			/* 11101100 */ {{7, 6, 3, 2, 5, 4, 1, 0}, false, 0b00011111},
			/* 11101101 */ {{3, 2, 7, 6, 1, 0, 5, 4}, true, 0b01101111},
			/* 11101110 */ {{3, 7, 1, 5, 2, 6, 0, 4}, true, 0b00111111},
			/* 11101111 */ {{3, 2, 1, 0, 7, 6, 5, 4}, false, 0b01111111},
			/* 11110000 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00001111},
			/* 11110001 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00011111},
			/* 11110010 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b00011111},
			/* 11110011 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b00111111},
			/* 11110100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00011111},
			/* 11110101 */ {{4, 6, 5, 7, 0, 2, 1, 3}, false, 0b00111111},
			/* 11110110 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b01101111},
			/* 11110111 */ {{4, 5, 6, 7, 0, 1, 2, 3}, true, 0b01111111},
			/* 11111000 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b00011111},
			/* 11111001 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b01101111},
			/* 11111010 */ {{5, 7, 4, 6, 1, 3, 0, 2}, true, 0b00111111},
			/* 11111011 */ {{5, 4, 7, 6, 1, 0, 3, 2}, false, 0b01111111},
			/* 11111100 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b00111111},
			/* 11111101 */ {{6, 7, 4, 5, 2, 3, 0, 1}, false, 0b01111111},
			/* 11111110 */ {{7, 6, 5, 4, 3, 2, 1, 0}, true, 0b01111111},
			/* 11111111 */ {{0, 1, 2, 3, 4, 5, 6, 7}, false, 0b11111111},
		};

	if (a.n->is_leaf() && b.n->is_leaf() && c.n->is_leaf() && d.n->is_leaf() && aa.n->is_leaf() && ba.n->is_leaf() && ca.n->is_leaf() && da.n->is_leaf())
	{
		int index = 0;
		TNode* n[8] = { a.n, b.n, c.n, d.n, aa.n, ba.n, ca.n, da.n };
		for (int i = 0; i < 8; i++)
		{
			if (sign(n[i]->node) > 0)
			{
				index += 1 << i;
			}
		}
		const auto& proc = table[index];
		if (proc.code == 0b00000000 || proc.code == 0b11111111)
		{
			return false;
		}
		std::array<TNode*, 8> trans_vertices;
		for (int i = 0; i < 8; i++)
		{
			trans_vertices[i] = n[proc.trans[i]];
		}
		std::array<Eigen::Vector3f, 12> points;
		
		auto calculate_point = [&](int e_index, int v_index1, int v_index2) {
			auto& v1 = *trans_vertices[v_index1];
			auto& v2 = *trans_vertices[v_index2];
			
			if ((v1.node[3] > 0 ? 1 : -1) != (v2.node[3] > 0 ? 1 : -1))
			{
				Eigen::Vector4f tmpv1 = v1.node, tmpv2 = v2.node, tmpv = Eigen::Vector4f::Zero();
				Eigen::Vector3f tmpg = Eigen::Vector3f::Zero();
				float ratio;
				while ((tmpv1 - tmpv2).head(3).norm() > 
				(IS_CONST_RADIUS ? constructor->getConstRadius() : constructor->getSearcher()->getAvgRadius()) / constructor->getInfluenceFactor())
				{
					tmpv[0] =  (tmpv1[0] + tmpv2[0]) / 2;
					tmpv[1] =  (tmpv1[1] + tmpv2[1]) / 2;
					tmpv[2] =  (tmpv1[2] + tmpv2[2]) / 2;
					constructor->getEvaluator()->SingleEval(tmpv.head(3), tmpv[3], tmpg);
					if ((tmpv[3] > 0 ? 1 : -1) == (tmpv1[3] > 0 ? 1 : -1))
					{
						tmpv1 = tmpv;
						tmpv.setZero();
					}
					else if ((tmpv[3] > 0 ? 1 : -1) == (tmpv2[3] > 0 ? 1 : -1))
					{
						tmpv2 = tmpv;
						tmpv.setZero();
					}
				}
				ratio = invlerp(tmpv1[3], tmpv2[3], 0.0f);
				if (ratio < constructor->getRatioTolerance())
					tmpv = tmpv1;
				else if (ratio > (1 - constructor->getRatioTolerance()))
					tmpv = tmpv2;
				else
					tmpv = lerp(tmpv1, tmpv2, ratio);

				points[e_index] = tmpv.head(3);

				// constructor->getEvaluator()->SingleEval(tmpv.head(3), tmpv[3], tmpg);
				// while (abs(tmpv[3]) > 0.1f)
				// {
				// 	if ((tmpv[3] > 0 ? 1 : -1) == (tmpv1[3] > 0 ? 1 : -1))
				// 	{
				// 		tmpv1 = tmpv;
				// 		tmpv.setZero();
				// 	}
				// 	else if ((tmpv[3] > 0 ? 1 : -1) == (tmpv2[3] > 0 ? 1 : -1))
				// 	{
				// 		tmpv2 = tmpv;
				// 		tmpv.setZero();
				// 	}

				// 	// ratio = invlerp(tmpv1[3], tmpv2[3], 0.0f);
				// 	// if (ratio < constructor->getRatioTolerance())
				// 	// 	tmpv = tmpv1;
				// 	// else if (ratio > (1 - constructor->getRatioTolerance()))
				// 	// 	tmpv = tmpv2;
				// 	// else
				// 	tmpv = lerp(tmpv1, tmpv2, ratio);
				// 	constructor->getEvaluator()->SingleEval(tmpv.head(3), tmpv[3], tmpg);
				// }
				// points[e_index] = tmpv.head(3);
			}
		};

		calculate_point(0, 0, 1);
		calculate_point(1, 2, 3);
		calculate_point(2, 4, 5);
		calculate_point(3, 6, 7);
		calculate_point(4, 0, 2);
		calculate_point(5, 1, 3);
		calculate_point(6, 4, 6);
		calculate_point(7, 5, 7);
		calculate_point(8, 0, 4);
		calculate_point(9, 1, 5);
		calculate_point(10, 2, 6);
		calculate_point(11, 3, 7);

		auto append = [&](int index1, int index2, int index3) {
			const auto& p1 = points[index1];
			const auto& p2 = points[index2];
			const auto& p3 = points[index3];
			int pIdx1 = m->insert_vert(p1);
			int pIdx2 = m->insert_vert(p2);
			int pIdx3 = m->insert_vert(p3);
			if (proc.flip)
			{
				m->insert_tri(pIdx1, pIdx2, pIdx3);
			}
			else
			{
				m->insert_tri(pIdx3, pIdx2, pIdx1);
			}
		};

		switch (proc.code)
		{
		case 0b00000001:
			append(0, 4, 8);
			break;
		case 0b00000011:
			append(4, 8, 9);
			append(5, 4, 9);
			break;
		case 0b00000110:
			append(0, 9, 4);
			append(4, 9, 10);
			append(10, 9, 5);
			append(10, 5, 1);
			break;
		case 0b00000111:
			append(8, 9, 10);
			append(1, 10, 9);
			append(5, 1, 9);
			break;
		case 0b00001111:
			append(8, 9, 10);
			append(9, 11, 10);
			break;
		case 0b00010110:
			append(0, 8, 4);
			append(1, 10, 5);
			append(5, 10, 9);
			append(9, 10, 2);
			append(2, 10, 6);
			break;
		case 0b00010111:
			append(1, 10, 5);
			append(5, 10, 9);
			append(2, 9, 10);
			append(2, 10, 6);
			break;
		case 0b00011000:
			append(1, 5, 11);
			append(2, 8, 6);
			break;
		case 0b00011001:
			append(1, 4, 6);
			append(1, 6, 11);
			append(11, 6, 2);
			append(11, 2, 5);
			append(5, 2, 0);
			break;
		case 0b00011011:
			append(2, 9, 6);
			append(6, 9, 11);
			append(6, 11, 1);
			append(1, 4, 6);
			break;
		case 0b00011110:
			append(0, 8, 4);
			append(2, 9, 6);
			append(6, 9, 10);
			append(10, 9, 11);
			break;
		case 0b00011111:
			append(2, 9, 6);
			append(6, 9, 10);
			append(10, 9, 11);
			break;
		case 0b00111100:
			append(4, 5, 8);
			append(8, 5, 9);
			append(10, 6, 11);
			append(11, 6, 7);
			break;
		case 0b00111101:
			append(6, 7, 10);
			append(10, 7, 11);
			append(0, 5, 9);
			break;
		case 0b00111111:
			append(6, 7, 10);
			append(10, 7, 11);
			break;
		case 0b01101001:
			append(0, 5, 9);
			append(1, 4, 10);
			append(2, 6, 8);
			append(3, 7, 11);
			break;
		case 0b01101011:
			append(1, 4, 8);
			append(1, 8, 11);
			append(11, 8, 7);
			append(7, 8, 2);
			append(3, 6, 10);
			break;
		case 0b01101111:
			append(2, 6, 8);
			append(3, 7, 11);
			break;
		case 0b01111110:
			append(0, 8, 4);
			append(3, 7, 11);
			break;
		case 0b01111111:
			append(3, 7, 11);
			break;
		}
		return false;
	}
	return true;
}

bool VisitorExtract::on_node(TraversalData &td)
{
	if (constrained_depth != 0)
	{
		if ((td.depth == constrained_depth || td.n->is_leaf()))
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	else if (part != nullptr)
	{
		return false;
	} 
	else
	{
		return !td.n->is_leaf();
	}
}

bool VisitorExtract::on_edge(TraversalData& td00, TraversalData& td10, TraversalData& td01, TraversalData& td11)
{
	return !(td00.n->is_leaf() && td10.n->is_leaf() && td01.n->is_leaf() && td11.n->is_leaf());
}

bool VisitorExtract::on_face(TraversalData &td0, TraversalData &td1, char orient)
{
	if (constrained_depth != 0)
	{
		if ((td0.depth == constrained_depth || td0.n->is_leaf()) && (td1.depth == constrained_depth || td1.n->is_leaf()))
		{
			g->appendEdge(td0.n->nId, td1.n->nId);
			return false;
		}
		return true;
	}
	else
	{
		return !(td0.n->is_leaf() && td1.n->is_leaf());
	}
}
