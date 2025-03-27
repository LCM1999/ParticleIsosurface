#include <extract_manager.h>

namespace iso{


// Make sure at least one node has children
HOST_DEVICE bool ExtractManager::on_node(int i){
    return iso_octreeNs->childOffsets[i];
}

// Make sure at least one node has children
HOST_DEVICE bool ExtractManager::on_edge(int i00, int i10, int i01, int i11){
    return iso_octreeNs->childOffsets[i00] || iso_octreeNs->childOffsets[i10] || iso_octreeNs->childOffsets[i01] || iso_octreeNs->childOffsets[i11];
}

// Make sure at least one node has children
HOST_DEVICE bool ExtractManager::on_face(int i0, int i1){
    return iso_octreeNs->childOffsets[i0] || iso_octreeNs->childOffsets[i1];
}

/**
 * @brief copy current node's sub node to the assigned idx, according to the Index i for offset
 * 
 * @param assigned_idx 
 * @param curr_node_idx the childOffsets index, which is the current node
 * @param i             Index class --> the offset for the sub node
 * @return HOST_DEVICE 
 */
HOST_DEVICE void ExtractManager::copy_sub_node(int& assigned_idx, int curr_node_idx, Index i){
    int sub_node_begin = iso_octreeNs->childOffsets[curr_node_idx];
    if(sub_node_begin != 0){ // it is not leaf node
        int offset = i;
        assigned_idx = sub_node_begin + offset;
    }
    else{
        assigned_idx = curr_node_idx;
    }
    
}

}