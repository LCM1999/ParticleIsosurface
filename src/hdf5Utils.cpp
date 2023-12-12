#include <filesystem>
#include <iostream>

#include "hdf5Utils.hpp"


bool readShonDyParticleXDMF(const std::string dir_path,
                            const std::string xdmf_file,
                            std::vector<std::string> &files,
                            const int target_frame)
{
    std::string xdmf_path = dir_path + "/" + xdmf_file;
    if (!std::filesystem::exists(xdmf_path))
    {
        std::cout << "Error: cannot find xdmf file: " << xdmf_path << std::endl;
        return false;
    }

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xdmf_path.c_str());
    if (!result)
    {
        std::cout << "Load result: " << result.description() << std::endl;
        return false;
    }

    // prefix = doc.child("Xdmf").child("Domain").attribute("Name").as_string();
    int files_num = std::distance(doc.child("Xdmf").child("Domain").child("Grid").children("Grid").begin(), doc.child("Xdmf").child("Domain").child("Grid").children("Grid").end());
    int file_index = 0;
    std::string frame_path;
    for (pugi::xml_node frame: doc.child("Xdmf").child("Domain").child("Grid").children("Grid"))
    {
        file_index++;
        frame_path = std::string(frame.child("Geometry").child("DataItem").child_value());
        frame_path = frame_path.substr(0, frame_path.find_first_of(":"));
        // std::cout << frame_path << std::endl;
        if (target_frame > 0 && target_frame <= files_num)
        {
            if (target_frame == file_index)
            {
                files.push_back(frame_path);
            }
        } else {
            files.push_back(frame_path);
        }
    }
    return true;
}

void readInt(hid_t fileID, const std::string &vecLocation,
             std::vector<int> &readVec)
{
    auto dset = H5Dopen2(fileID, vecLocation.c_str(), H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            readVec.data());
    H5Dclose(dset);
}

void readDouble(hid_t fileID, const std::string &vecLocation,
                std::vector<double> &readVec)
{
    auto dset = H5Dopen2(fileID, vecLocation.c_str(), H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            readVec.data());
    H5Dclose(dset);
}

int size(hid_t fileID, const std::string &groupName)
{
    auto dsetID = H5Dopen(fileID, groupName.c_str(), H5P_DEFAULT);
    auto dspaceID = H5Dget_space(dsetID);
    const int ndims = H5Sget_simple_extent_ndims(dspaceID);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(dspaceID, dims.data(), NULL);
    H5Dclose(dsetID);
    return dims[0];
}