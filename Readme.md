Set path for HDF5 and Eigen in CMakeList.txt:  
    $\qquad$ set(EIGEN3_DIR C:/Lib/Eigen-3.4.0-MSVC/share/eigen3/cmake)  
    $\qquad$ set(HDF5_DIR C:/Lib/HDF5-1.10.7-MSVC/cmake/hdf5)  
Please replace the path by your local path.

If you want to run this program, you need to do the following:  
    $\qquad$ 1. Save shonDy's result as hdf5 files sequence in a dir, and make sure it contains "position" and "numberDensity".  
    $\qquad$ 2. Write a controlData.json, and put it in the hdf5 dir.  
    $\qquad$ 3. Write the file names of all frames to be batched in "H5_PATH", separate them with ",".  
    $\qquad$ 4. If you want to save the record file while running, you should write "NEED_RECORD" as true.  
    $\qquad$ 5. Always make sure the value of "P_RADIUS" is correct.  
    $\qquad$ 6. .obj files and record files will be output to the directory where the controlData.json is located.  
    $\qquad$ 8. In the terminal, enter the following command:  
    $\qquad$ $\qquad$     ./Isosurface_smiplify.exe [Path to controlData.json]  
    $\qquad$ 9. If you got "METIS ERROR, USE DEFAULT SINGLE THREAD" during a run, that is because the parallelization modification of Mesh Generate phase has not been completed, you will end up with normal results.