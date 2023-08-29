#include <filesystem>
#include <iostream>

#include "hdf5Utils.hpp"


bool readShonDyParticleXDMF(const std::string dir_path,
                            const std::string xdmf_file,
                            std::vector<std::string> &files)
{
    std::string xdmf_path = dir_path + "\\" + xdmf_file;
    if (!std::filesystem::exists(xdmf_path))
    {
        std::cout << "Error: cannot find xdmf file: " << xdmf_path << std::endl;
        return false;
    }

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xdmf_path);
    if (!result)
    {
        std::cout << "Load result: " << result.description() << std::endl;
        return false;
    }

    // prefix = doc.child("Xdmf").child("Domain").attribute("Name").as_string();

    for (pugi::xml_node frame: doc.child("Xdmf").child("Domain").child("Grid").children("Grid"))
    {
        // std::cout << frame.child("Time").attribute("Value").as_string() << std::endl;
        files.push_back(std::string(frame.child("Time").attribute("Value").as_string()) + ".h5");
    }
    return true;
}

bool readShonDyParticleData(const std::string &fileName,
                            std::vector<Eigen::Vector3f> &positions,
                            std::vector<float>* radiuses, const float scale)
{
    if (!std::filesystem::exists(fileName))
    {
        std::cout << "Error: cannot find hdf5 file: " << fileName << std::endl;
        return false;
    }

    // open file
    auto hdf5FileID = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (hdf5FileID < 0)
    {
        std::cout << "Error: cannot read hdf5 file: " << fileName << std::endl;
        return false;
    }

    // read particle positions
    auto nodeSize = size(hdf5FileID, "position");
    const int numberOfNodes = nodeSize / 3;
    std::vector<double> nodes(nodeSize);
    std::vector<double> radiusesDouble(numberOfNodes);
    readDouble(hdf5FileID, "position", nodes);
    positions.resize(numberOfNodes);

    // read particle densities
    if (radiuses != nullptr)
    {
        readDouble(hdf5FileID, "particleRadius", radiusesDouble);
        radiuses->resize(numberOfNodes);
    }
    
    // convert double data to float
    for (int i = 0; i < numberOfNodes; i++)
    {
        positions[i] =
            Eigen::Vector3f(nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2]);
        if (radiuses != nullptr)
        {
            (*radiuses)[i] = radiusesDouble[i] * scale;
        }
    }

    // close file
    H5Fclose(hdf5FileID);
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