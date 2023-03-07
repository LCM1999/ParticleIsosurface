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

bool VisitorExtract::on_vert(TraversalData& a, TraversalData& b, TraversalData& c, TraversalData& d, TraversalData& aa, TraversalData& ba, TraversalData& ca, TraversalData& da)
{
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
				(IS_CONST_RADIUS ? constructor->getConstRadius() : constructor->getSearcher()->getMinRadius()) / 2)
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
					} else {
						break;
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
	return !td.n->is_leaf();
}

bool VisitorExtract::on_edge(TraversalData& td00, TraversalData& td10, TraversalData& td01, TraversalData& td11)
{
	return !(td00.n->is_leaf() && td10.n->is_leaf() && td01.n->is_leaf() && td11.n->is_leaf());
}

bool VisitorExtract::on_face(TraversalData &td0, TraversalData &td1, char orient)
{
	return !(td0.n->is_leaf() && td1.n->is_leaf());
}
