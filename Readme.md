# Isosurface generator
## Requirements
* HDF5
* Eigen3
* OpenMP & MPI

Please make sure to set paths for Eigen3 and HDF5 in CMakeLists.txt.
```bash
set(EIGEN3_DIR C:/Lib/Eigen-3.4.0-MSVC/share/eigen3/cmake)
set(HDF5_DIR C:/Lib/HDF5-1.10.7-MSVC/cmake/hdf5)
```

## Data Preparation & Settings
1. Create a case dir `$CASE_DIR$` and save shonDy's particle results in it (one `hdf5` file for each frame).  
2. Create a controlData.json in `$CASE_DIR$`.  
3. Set `H5_PATH` as the file names for all frames, separated by ",".  
4. Set `NEED_RECORD` as true if you need to save the record files while running (otherwise set false).  
5. Set `P_RADIUS` as the particle radius (always make sure it's correct).  
## Build & Run

1. To build the program, please enter
```bash
./Allmake
``` 
It is built in release mode by default. If you want to build it in debug mode, please enter
```bash
./Allmake debug
``` 
2. To run the program, please enter 
```bash
./Isosurface_smiplify.exe $CASE_DIR$
```
3. The isosurface mesh results will be exported as `obj` files to `$CASE_DIR$`.
4. If you got `METIS ERROR, USE DEFAULT SINGLE THREAD` during a run, that is because the parallelization modification of Mesh Generate phase has not been completed, you will end up with normal results.