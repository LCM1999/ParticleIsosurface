#pragma once

namespace iso{

// enum TraversalType
// {
//     trav_node,
// 	trav_face,
// 	trav_edge,
// 	trav_vert
// };

// template <TraversalType TT, class EM> 
// void traverse_vert_CPU(EM &em, int n000, int n100, int n010, int n110, int n001, int n101, int n011, int n111) //nXYZ
// {
// 	if (!em.on_vert_CPU(n000, n100, n010, n110, n001, n101, n011, n111))
// 		return;
	
// 	int leaves[8];
// 	em.copy_sub_node(leaves[Index(0, 0, 0)], n000, Index(1, 1, 1));
// 	em.copy_sub_node(leaves[Index(1, 0, 0)], n100, Index(0, 1, 1));
// 	em.copy_sub_node(leaves[Index(0, 1, 0)], n010, Index(1, 0, 1));
// 	em.copy_sub_node(leaves[Index(1, 1, 0)], n110, Index(0, 0, 1));
// 	em.copy_sub_node(leaves[Index(0, 0, 1)], n001, Index(1, 1, 0));
// 	em.copy_sub_node(leaves[Index(1, 0, 1)], n101, Index(0, 1, 0));
// 	em.copy_sub_node(leaves[Index(0, 1, 1)], n011, Index(1, 0, 0));
// 	em.copy_sub_node(leaves[Index(1, 1, 1)], n111, Index(0, 0, 0));

// 	traverse_vert_CPU<TT,EM>(em, leaves[Index(0, 0, 0)],
// 							 leaves[Index(1, 0, 0)],
// 							 leaves[Index(0, 1, 0)],
// 							 leaves[Index(1, 1, 0)],
// 							 leaves[Index(0, 0, 1)],
// 							 leaves[Index(1, 0, 1)],
// 							 leaves[Index(0, 1, 1)],
// 							 leaves[Index(1, 1, 1)]);

// }

// template <TraversalType TT, class EM>
// void traverse_edge_x_CPU(EM &em, int n00, int n10, int n01, int n11) //nYZ
// {
// 	if (!em.on_edge(n00, n10, n01, n11))
// 		return;

// 	int leaves[8];
// 	for (int i = 0; i < 2; i++)
// 	{
// 		em.copy_sub_node(leaves[Index(i, 0, 0)], n00, Index(i, 1, 1));
// 		em.copy_sub_node(leaves[Index(i, 1, 0)], n10, Index(i, 0, 1));
// 		em.copy_sub_node(leaves[Index(i, 0, 1)], n01, Index(i, 1, 0));
// 		em.copy_sub_node(leaves[Index(i, 1, 1)], n11, Index(i, 0, 0));
// 	}

// 	for (int i = 0; i < 2; i++)
// 	{
// 		traverse_edge_x_CPU<TT,EM>(em, leaves[Index(i, 0, 0)], leaves[Index(i, 1, 0)], leaves[Index(i, 0, 1)], leaves[Index(i, 1, 1)]);
// 	}

// 	if (TT >= trav_vert)
// 	{
// 		traverse_vert_CPU<TT,EM>(em, leaves[Index(0, 0, 0)],
// 								 leaves[Index(1, 0, 0)],
// 								 leaves[Index(0, 1, 0)],
// 								 leaves[Index(1, 1, 0)],
// 								 leaves[Index(0, 0, 1)],
// 								 leaves[Index(1, 0, 1)],
// 								 leaves[Index(0, 1, 1)],
// 								 leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_edge_y_CPU(EM &em, int n00, int n10, int n01, int n11) //nXZ
// {
// 	if (!em.on_edge(n00, n10, n01, n11))
// 		return;

// 	int leaves[8];
// 	for (int i = 0; i < 2; i++)
// 	{
// 		em.copy_sub_node(leaves[Index(0, i, 0)], n00, Index(1, i, 1));
// 		em.copy_sub_node(leaves[Index(1, i, 0)], n10, Index(0, i, 1));
// 		em.copy_sub_node(leaves[Index(0, i, 1)], n01, Index(1, i, 0));
// 		em.copy_sub_node(leaves[Index(1, i, 1)], n11, Index(0, i, 0));
// 	}

// 	for (int i = 0; i < 2; i++)
// 	{
// 		traverse_edge_y_CPU<TT,EM>(em, leaves[Index(0, i, 0)], leaves[Index(1, i, 0)], leaves[Index(0, i, 1)], leaves[Index(1, i, 1)]);
// 	}

// 	if (TT >= trav_vert)
// 	{
// 		traverse_vert_CPU<TT,EM>(em, leaves[Index(0, 0, 0)],
// 								 leaves[Index(1, 0, 0)],
// 								 leaves[Index(0, 1, 0)],
// 								 leaves[Index(1, 1, 0)],
// 								 leaves[Index(0, 0, 1)],
// 								 leaves[Index(1, 0, 1)],
// 								 leaves[Index(0, 1, 1)],
// 								 leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_edge_z_CPU(EM &em, int n00, int n10, int n01, int n11) //nXY
// {
// 	if (!em.on_edge(n00, n10, n01, n11))
// 		return;

// 	int leaves[8];
// 	for (int i = 0; i < 2; i++)
// 	{
// 		em.copy_sub_node(leaves[Index(0, 0, i)], n00, Index(1, 1, i));
// 		em.copy_sub_node(leaves[Index(1, 0, i)], n10, Index(0, 1, i));
// 		em.copy_sub_node(leaves[Index(0, 1, i)], n01, Index(1, 0, i));
// 		em.copy_sub_node(leaves[Index(1, 1, i)], n11, Index(0, 0, i));
// 	}

// 	for (int i = 0; i < 2; i++)
// 	{
// 		traverse_edge_z_CPU<TT,EM>(em, leaves[Index(0, 0, i)], leaves[Index(1, 0, i)], leaves[Index(0, 1, i)], leaves[Index(1, 1, i)]);
// 	}

// 	if (TT >= trav_vert)
// 	{
// 		traverse_vert_CPU<TT,EM>(em, leaves[Index(0, 0, 0)],
// 								 leaves[Index(1, 0, 0)],
// 								 leaves[Index(0, 1, 0)],
// 								 leaves[Index(1, 1, 0)],
// 								 leaves[Index(0, 0, 1)],
// 								 leaves[Index(1, 0, 1)],
// 								 leaves[Index(0, 1, 1)],
// 								 leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_face_x_CPU(EM& em, int n0, int n1){
// 	if(!em.on_face(n0, n1))
// 		return;
	
// 	int leaves[8];
// 	for(Index i = 0; i < 4; i++){
// 		em.copy_sub_node(leaves[Index(0, i.x, i.y)], n0, Index(1, i.x, i.y));
// 		em.copy_sub_node(leaves[Index(1, i.x, i.y)], n1, Index(0, i.x, i.y));
// 	}

// 	for(Index i = 0; i < 4; i++){
// 		traverse_face_x_CPU<TT, EM>(em, leaves[Index(0, i.x, i.y)], leaves[Index(1, i.x, i.y)]);
// 	}

// 	if(TT >= trav_edge){
// 		for(int i = 0; i < 2; i++){
// 			traverse_edge_y_CPU<TT, EM>(em, leaves[Index(0, i, 0)], leaves[Index(1, i, 0)], leaves[Index(0, i, 1)], leaves[Index(1, i, 1)]);
// 			traverse_edge_z_CPU<TT, EM>(em, leaves[Index(0, 0, i)], leaves[Index(1, 0, i)], leaves[Index(0, 1, i)], leaves[Index(1, 1, i)]);
// 		}
// 	}

// 	if(TT >= trav_vert){
// 		traverse_vert_CPU<TT, EM>(em, leaves[Index(0, 0, 0)], 
// 									leaves[Index(1, 0, 0)], 
// 									leaves[Index(0, 1, 0)], 
// 									leaves[Index(1, 1, 0)], 
// 									leaves[Index(0, 0, 1)], 
// 									leaves[Index(1, 0, 1)], 
// 									leaves[Index(0, 1, 1)], 
// 									leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_face_y_CPU(EM& em, int n0, int n1){
// 	if(!em.on_face(n0, n1))
// 		return;
	
// 	int leaves[8];
// 	for(Index i = 0; i < 4; i++){
// 		em.copy_sub_node(leaves[Index(i.x, 0, i.y)], n0, Index(i.x, 1, i.y));
// 		em.copy_sub_node(leaves[Index(i.x, 1, i.y)], n1, Index(i.x, 0, i.y));
// 	}

// 	for(Index i = 0; i < 4; i++){
// 		traverse_face_y_CPU<TT, EM>(em, leaves[Index(i.x, 0, i.y)], leaves[Index(i.x, 1, i.y)]);
// 	}

// 	if(TT >= trav_edge){
// 		for(int i = 0; i < 2; i++){
// 			traverse_edge_x_CPU<TT, EM>(em, leaves[Index(i, 0, 0)], leaves[Index(i, 1, 0)], leaves[Index(i, 0, 1)], leaves[Index(i, 1, 1)]);
// 			traverse_edge_z_CPU<TT, EM>(em, leaves[Index(0, 0, i)], leaves[Index(1, 0, i)], leaves[Index(0, 1, i)], leaves[Index(1, 1, i)]);
// 		}
// 	}

// 	if(TT >= trav_vert){
// 		traverse_vert_CPU<TT, EM>(em, leaves[Index(0, 0, 0)], 
// 									leaves[Index(1, 0, 0)], 
// 									leaves[Index(0, 1, 0)], 
// 									leaves[Index(1, 1, 0)], 
// 									leaves[Index(0, 0, 1)], 
// 									leaves[Index(1, 0, 1)], 
// 									leaves[Index(0, 1, 1)], 
// 									leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_face_z_CPU(EM& em, int n0, int n1){
// 	if(!em.on_face(n0, n1))
// 		return;
	
// 	int leaves[8];
// 	for(Index i = 0; i < 4; i++){
// 		em.copy_sub_node(leaves[Index(i.x, i.y, 0)], n0, Index(i.x, i.y, 1));
// 		em.copy_sub_node(leaves[Index(i.x, i.y, 1)], n1, Index(i.x, i.y, 0));
// 	}

// 	for(Index i = 0; i < 4; i++){
// 		traverse_face_z_CPU<TT, EM>(em, leaves[Index(i.x, i.y, 0)], leaves[Index(i.x, i.y, 1)]);
// 	}

// 	if(TT >= trav_edge){
// 		for(int i = 0; i < 2; i++){
// 			traverse_edge_x_CPU<TT, EM>(em, leaves[Index(i, 0, 0)], leaves[Index(i, 1, 0)], leaves[Index(i, 0, 1)], leaves[Index(i, 1, 1)]);
// 			traverse_edge_y_CPU<TT, EM>(em, leaves[Index(0, i, 0)], leaves[Index(1, i, 0)], leaves[Index(0, i, 1)], leaves[Index(1, i, 1)]);
// 		}
// 	}

// 	if(TT >= trav_vert){
// 		traverse_vert_CPU<TT, EM>(em, leaves[Index(0, 0, 0)], 
// 									leaves[Index(1, 0, 0)], 
// 									leaves[Index(0, 1, 0)], 
// 									leaves[Index(1, 1, 0)], 
// 									leaves[Index(0, 0, 1)], 
// 									leaves[Index(1, 0, 1)], 
// 									leaves[Index(0, 1, 1)], 
// 									leaves[Index(1, 1, 1)]);
// 	}

// }

// template <TraversalType TT, class EM>
// void traverse_node_CPU(EM& em, int n){
// 	if(!em.on_node(n))
// 		return;
	
// 	int leaves[8];
// 	for(Index i = 0; i < 8; i++){
// 		em.copy_sub_node(leaves[i], n, i);
// 		traverse_node_CPU<TT, EM>(em, leaves[i]);
// 	}

// 	if(TT >= trav_face){
// 		for(Index i = 0; i < 4; i++){
// 			traverse_face_x_CPU<TT, EM>(em, leaves[Index(0, i.x, i.y)], leaves[Index(1, i.x, i.y)]);
// 			traverse_face_y_CPU<TT, EM>(em, leaves[Index(i.x, 0, i.y)], leaves[Index(i.x, 1, i.y)]);
// 			traverse_face_z_CPU<TT, EM>(em, leaves[Index(i.x, i.y, 0)], leaves[Index(i.x, i.y, 1)]);
// 		}
// 	}

// 	if(TT >= trav_edge){
// 		for(int i = 0; i < 2; i++){
// 			traverse_edge_x_CPU<TT, EM>(em, leaves[Index(i, 0, 0)], leaves[Index(i, 1, 0)], leaves[Index(i, 0, 1)], leaves[Index(i, 1, 1)]);
// 			traverse_edge_y_CPU<TT, EM>(em, leaves[Index(0, i, 0)], leaves[Index(1, i, 0)], leaves[Index(0, i, 1)], leaves[Index(1, i, 1)]);
// 			traverse_edge_z_CPU<TT, EM>(em, leaves[Index(0, 0, i)], leaves[Index(1, 0, i)], leaves[Index(0, 1, i)], leaves[Index(1, 1, i)]);
// 		}
// 	}

// 	if(TT >= trav_vert){
// 		traverse_vert_CPU<TT, EM>(em, leaves[Index(0, 0, 0)], 
// 									leaves[Index(1, 0, 0)], 
// 									leaves[Index(0, 1, 0)], 
// 									leaves[Index(1, 1, 0)], 
// 									leaves[Index(0, 0, 1)], 
// 									leaves[Index(1, 0, 1)], 
// 									leaves[Index(0, 1, 1)], 
// 									leaves[Index(1, 1, 1)]);
// 	}
// }

}