If you build this project with cmake, you need to modify the MEDIS path in CMakeList.

If you wanna run this program, you need to do the following:  
    $\qquad$ 1. Save shonDy's result as csv files sequence in a dir and only retain only "number density", you can do this with paraview easily.  
    $\qquad$ 2. Write a controlData.json, and put it in the csv dir.  
    $\qquad$ 3. Write the file names of all frames to be batched in "CSV_PATH", separate them with ",".  
    $\qquad$ 4. If you export csv with paraview, you should write "CSV_TYPE" as 1,
        if you export csv with shonDy, you should write "CSV_TYPE" as 2.  
    $\qquad$ 5. If you want to save the record file while running, you should write "NEED_RECORD" as true.  
    $\qquad$ 6. Always make sure the value of "P_RADIUS" is correct.  
    $\qquad$ 7. .obj files and record files will be output to the directory where the controlData.json is located.  
    $\qquad$ 8. In the terminal, enter the following command:  
    $\qquad$ $\qquad$     ./Isosurface_smiplify.exe [Path to controlData.json]  
    $\qquad$ 9. If you got "METIS ERROR, USE DEFAULT SINGLE THREAD" during a run, that is because the parallelization modification of Mesh Generate phase has not been completed, you will end up with normal results.

