#include <extract_manager.h>


namespace iso{

// bool ExtractManager::on_vert_CPU(int a, int b, int c, int d, int aa, int ba, int ca, int da){
//     if(!iso_octreeNs->childOffsets[a] && !iso_octreeNs->childOffsets[b] && !iso_octreeNs->childOffsets[c] && !iso_octreeNs->childOffsets[d] && !iso_octreeNs->childOffsets[aa] && !iso_octreeNs->childOffsets[ba] && !iso_octreeNs->childOffsets[ca] && !iso_octreeNs->childOffsets[da]){
//         int index = 0;
//         std::vector<int> leafIdxs = {iso_octreeNs->internalToLeaf[a], iso_octreeNs->internalToLeaf[b], 
//                                     iso_octreeNs->internalToLeaf[c], iso_octreeNs->internalToLeaf[d], 
//                                     iso_octreeNs->internalToLeaf[aa], iso_octreeNs->internalToLeaf[ba], 
//                                     iso_octreeNs->internalToLeaf[ca], iso_octreeNs->internalToLeaf[da]};
//         for(int i = 0; i < 8; i++){
//             if(sign(iso_octreeNs->scalars[leafIdxs[i]]) > 0){
//                 index += 1 << i;
//             }
//         }
//         DualGrid dual_grid;
//         const auto& proc = table[index];
//         for(int i = 0; i < 8; i++){
//             dual_grid.leafIdxs[i] = leafIdxs[proc.trans[i]];
//             dual_grid.proc = proc;
//         }
//         if (!(proc.code == 0b00000000 || proc.code == 0b11111111))
// 		{
//             dual_grids.push_back(dual_grid);
// 		}
//         return false;
//     }
//     else{
//         return true;
//     }
// }

// void ExtractManager::cal_vertices_CPU(){
//     dual_cells.resize(dual_grids.size());
//     #pragma omp parallel for
//     for(size_t i = 0; i < dual_grids.size(); i++){
//         auto calculate_point = [&](int e_index, int v_index1, int v_index2){
//             auto& v1 = dual_grids[i].nodeIdx[v_index1];
//             auto& v2 = dual_grids[i].nodeIdx[v_index2];

//             float scalar_v1 = iso_octreeNs->scalars[iso_octreeNs->internalToLeaf[v1]];
//             float scalar_v2 = iso_octreeNs->scalars[iso_octreeNs->internalToLeaf[v2]];
//             if(scalar_v1 * scalar_v2 < 0){
//                 Vec3f tmpv;
//                 Vec3f tmpv1 = iso_octreeNs->centers[v1];
//                 Vec3f tmpv2 = iso_octreeNs->centers[v2];
//                 float d = std::sqrt((tmpv1.x - tmpv2.x) * (tmpv1.x - tmpv2.x) + (tmpv1.y - tmpv2.y) * (tmpv1.y - tmpv2.y) + (tmpv1.z - tmpv2.z) * (tmpv1.z - tmpv2.z));
//                 float r = (IS_CONST_RADIUS ? RADIUS : MIN_RADIUS) / 2;
//                 while(d > r){
//                     tmpv.x = (tmpv1.x + tmpv2.x) / 2;
//                     tmpv.y = (tmpv1.y + tmpv2.y) / 2;
//                     tmpv.z = (tmpv1.z + tmpv2.z) / 2;
//                     float scalar_tmp = 0.0;
//                     constructor->getEvaluator()->SingleEvalCPU(tmpv, scalar_tmp);

//                     if(scalar_tmp * scalar_v1 >= 0){
//                         tmpv1 = tmpv;
//                         scalar_v1 = scalar_tmp;
//                         tmpv = Vec3f(0.0, 0.0, 0.0);
//                     }
//                     else if(scalar_tmp * scalar_v2 >= 0){
//                         tmpv2 = tmpv;
//                         scalar_v2 = scalar_tmp;
//                         tmpv = Vec3f(0.0, 0.0, 0.0);
//                     }
//                     else{
//                         break;
//                     }
//                     d /= 2;
//                 }
//                 float ratio =  (0.0f - scalar_v1) / (scalar_v2 - scalar_v1); // tmpv1[3] tmpv2[3] 0.0f
//                 if(ratio < 0.1){
//                     tmpv = tmpv1;
//                 }
//                 else if(ratio > 0.9){
//                     tmpv = tmpv2;
//                 }
//                 else{
//                     tmpv.x = tmpv1.x + ratio * (tmpv2.x - tmpv1.x);
//                     tmpv.y = tmpv1.y + ratio * (tmpv2.y - tmpv1.y);
//                     tmpv.z = tmpv1.z + ratio * (tmpv2.z - tmpv1.z);
//                 }
//                 dual_cells[i].vertices[e_index] = tmpv;
//             }
//         };

//         for (int j = 0; j < 12; j++)
// 		{
// 			// calculate_point(j, dual_edge2vert[j][0], dual_edge2vert[j][1]);
//             int v_index1 = dual_edge2vert[j][0];
//             int v_index2 = dual_edge2vert[j][1];
//             int e_index = j;
//             auto& leaf_idx1 = dual_grids[i].leafIdxs[v_index1];
//             auto& leaf_idx2 = dual_grids[i].leafIdxs[v_index2];

//             int leaf_idx1_to_internal = iso_octreeNs->leafToInternal[leaf_idx1];
//             int leaf_idx2_to_internal = iso_octreeNs->leafToInternal[leaf_idx2];

//             float scalar_v1 = iso_octreeNs->scalars[leaf_idx1];
//             float scalar_v2 = iso_octreeNs->scalars[leaf_idx2];
//             if(sign(scalar_v1) != sign(scalar_v2)){
//                 Vec3f tmpv;
//                 Vec3f tmpv1 = iso_octreeNs->centers[leaf_idx1_to_internal];
//                 Vec3f tmpv2 = iso_octreeNs->centers[leaf_idx2_to_internal];
//                 float d = std::sqrt((tmpv1.x - tmpv2.x) * (tmpv1.x - tmpv2.x) + (tmpv1.y - tmpv2.y) * (tmpv1.y - tmpv2.y) + (tmpv1.z - tmpv2.z) * (tmpv1.z - tmpv2.z));
//                 float r = (IS_CONST_RADIUS ? RADIUS : MIN_RADIUS) / 2;
//                 while(d > r){
//                     tmpv.x = (tmpv1.x + tmpv2.x) / 2;
//                     tmpv.y = (tmpv1.y + tmpv2.y) / 2;
//                     tmpv.z = (tmpv1.z + tmpv2.z) / 2;
//                     float scalar_tmp = 0.0;
//                     constructor->getEvaluator()->SingleEvalCPU(tmpv, scalar_tmp);

//                     if(sign(scalar_tmp) == sign(scalar_v1) ){
//                         tmpv1 = tmpv;
//                         scalar_v1 = scalar_tmp;
//                         tmpv = Vec3f(0.0, 0.0, 0.0);
//                     }
//                     else if(sign(scalar_tmp) == sign(scalar_v2)){
//                         tmpv2 = tmpv;
//                         scalar_v2 = scalar_tmp;
//                         tmpv = Vec3f(0.0, 0.0, 0.0);
//                     }
//                     else{
//                         break;
//                     }
//                     d /= 2;
//                 }
//                 float ratio =  (0.0f - scalar_v1) / (scalar_v2 - scalar_v1); // tmpv1[3] tmpv2[3] 0.0f
//                 if(ratio < 0.1){
//                     tmpv = tmpv1;
//                 }
//                 else if(ratio > 0.9){
//                     tmpv = tmpv2;
//                 }
//                 else{
//                     tmpv.x = tmpv1.x + ratio * (tmpv2.x - tmpv1.x);
//                     tmpv.y = tmpv1.y + ratio * (tmpv2.y - tmpv1.y);
//                     tmpv.z = tmpv1.z + ratio * (tmpv2.z - tmpv1.z);
//                 }
//                 dual_cells[i].vertices[e_index] = tmpv;
//             }
// 		}

// 		auto append = [&](int index1, int index2, int index3) {
// 			dual_cells[i].faces.push_back({index1, index2, index3});
// 		};

// 		switch (dual_grids[i].proc.code)
// 		{
// 		case 0b00000001:
// 			append(0, 4, 8);
// 			break;
// 		case 0b00000011:
// 			append(4, 8, 9);
// 			append(5, 4, 9);
// 			break;
// 		case 0b00000110:
// 			append(0, 9, 4);
// 			append(4, 9, 10);
// 			append(10, 9, 5);
// 			append(10, 5, 1);
// 			break;
// 		case 0b00000111:
// 			append(8, 9, 10);
// 			append(1, 10, 9);
// 			append(5, 1, 9);
// 			break;
// 		case 0b00001111:
// 			append(8, 9, 10);
// 			append(9, 11, 10);
// 			break;
// 		case 0b00010110:
// 			append(0, 8, 4);
// 			append(1, 10, 5);
// 			append(5, 10, 9);
// 			append(9, 10, 2);
// 			append(2, 10, 6);
// 			break;
// 		case 0b00010111:
// 			append(1, 10, 5);
// 			append(5, 10, 9);
// 			append(2, 9, 10);
// 			append(2, 10, 6);
// 			break;
// 		case 0b00011000:
// 			append(1, 5, 11);
// 			append(2, 8, 6);
// 			break;
// 		case 0b00011001:
// 			append(1, 4, 6);
// 			append(1, 6, 11);
// 			append(11, 6, 2);
// 			append(11, 2, 5);
// 			append(5, 2, 0);
// 			break;
// 		case 0b00011011:
// 			append(2, 9, 6);
// 			append(6, 9, 11);
// 			append(6, 11, 1);
// 			append(1, 4, 6);
// 			break;
// 		case 0b00011110:
// 			append(0, 8, 4);
// 			append(2, 9, 6);
// 			append(6, 9, 10);
// 			append(10, 9, 11);
// 			break;
// 		case 0b00011111:
// 			append(2, 9, 6);
// 			append(6, 9, 10);
// 			append(10, 9, 11);
// 			break;
// 		case 0b00111100:
// 			append(4, 5, 8);
// 			append(8, 5, 9);
// 			append(10, 6, 11);
// 			append(11, 6, 7);
// 			break;
// 		case 0b00111101:
// 			append(6, 7, 10);
// 			append(10, 7, 11);
// 			append(0, 5, 9);
// 			break;
// 		case 0b00111111:
// 			append(6, 7, 10);
// 			append(10, 7, 11);
// 			break;
// 		case 0b01101001:
// 			append(0, 5, 9);
// 			append(1, 4, 10);
// 			append(2, 6, 8);
// 			append(3, 7, 11);
// 			break;
// 		case 0b01101011:
// 			append(1, 4, 8);
// 			append(1, 8, 11);
// 			append(11, 8, 7);
// 			append(7, 8, 2);
// 			append(3, 6, 10);
// 			break;
// 		case 0b01101111:
// 			append(2, 6, 8);
// 			append(3, 7, 11);
// 			break;
// 		case 0b01111110:
// 			append(0, 8, 4);
// 			append(3, 7, 11);
// 			break;
// 		case 0b01111111:
// 			append(3, 7, 11);
// 			break;
// 		}


//     }
// }

// void ExtractManager::generate_mesh_CPU(){
    // int tmp_vert_ids[3] = {0};
	// for (size_t i = 0; i < dual_grids.size(); i++)
	// {
	// 	for (size_t j = 0; j < dual_cells[i].faces.size(); j++)
	// 	{
	// 		for (size_t k = 0; k < 3; k++)
	// 		{
    //             int leaf_node1_idx = dual_grids[i].leafIdxs[dual_edge2vert[dual_cells[i].faces[j][k]][0]];
    //             int leaf_node2_idx = dual_grids[i].leafIdxs[dual_edge2vert[dual_cells[i].faces[j][k]][1]];
    //             Vec3f pos3f = dual_cells[i].vertices[dual_cells[i].faces[j][k]];
    //             Eigen::Vector3f pos = Eigen::Vector3f(pos3f.x, pos3f.y, pos3f.z);
	// 			tmp_vert_ids[k] = m->insert_vert(
	// 				leaf_node1_idx,
	// 				leaf_node2_idx,
	// 				pos);
	// 		}
	// 		if (dual_grids[i].proc.flip)
	// 		{
	// 			m->insert_tri(tmp_vert_ids[0], tmp_vert_ids[1], tmp_vert_ids[2]);
	// 		}
	// 		else
	// 		{
	// 			m->insert_tri(tmp_vert_ids[2], tmp_vert_ids[1], tmp_vert_ids[0]);
	// 		}
	// 	}
    // }
// }

}