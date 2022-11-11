# Find Metis library
if (NOT (Metis_INCLUDE_DIRS AND Metis_LIBRARIES))
    if(WIN32)
        if(METIS_ROOT)
                find_path(Metis_INCLUDE_DIRS
                    NAMES metis.h 
                    PATHS "${METIS_ROOT}"
                    PATH_SUFFIXES metis include
                )

                find_library(METIS_LIBRARY metis
                    PATHS "${METIS_ROOT}"
                    PATH_SUFFIXES lib
                )
            set(Metis_LIBRARIES "${METIS_LIBRARY}")

        endif()
    else()
        if(METIS_ROOT)    
            find_path(Metis_INCLUDE_DIRS
                NAMES metis.h 
                PATHS ${METIS_ROOT}
                PATH_SUFFIXES metis include
                )

            find_library(METIS_LIBRARY metis
                PATHS ${METIS_ROOT}
                PATH_SUFFIXES lib
                )
            
            find_library(GKlib_LIBRARY GKlib
                PATHS ${METIS_ROOT}
                PATH_SUFFIXES lib
                )
            set(Metis_LIBRARIES "${METIS_LIBRARY};${GKlib_LIBRARY}")
        else()
            find_path(Metis_INCLUDE_DIRS
                NAMES metis.h 
                PATHS $ENV{METIS_ROOT}
                PATH_SUFFIXES metis include
                )

            find_library(METIS_LIBRARY metis
                PATHS $ENV{METIS_ROOT}
                PATH_SUFFIXES lib
                )

            set(Metis_LIBRARIES "${METIS_LIBRARY}")

        endif() 
    endif()
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Metis DEFAULT_MSG Metis_INCLUDE_DIRS Metis_LIBRARIES)

mark_as_advanced(Metis_INCLUDE_DIRS Metis_LIBRARIES)

if (Metis_FOUND)
    add_library(Metis_metis INTERFACE)
    add_library(Metis::metis ALIAS Metis_metis)
    target_include_directories(Metis_metis INTERFACE ${Metis_INCLUDE_DIRS})
    target_link_libraries(Metis_metis INTERFACE ${Metis_LIBRARIES})

    if (METIS_LIBRARY)
        target_compile_definitions(Metis_metis INTERFACE AMGCL_HAVE_METIS)
    endif()
endif()

