if(WIN32)
    set(FCOMPARE_EXE ${CMAKE_BINARY_DIR}/Submodules/AMReX/Tools/Plotfile/*/amrex_fcompare.exe CACHE INTERNAL "Path to fcompare executable for regression tests")
    set(FEXTREMA_EXE ${CMAKE_BINARY_DIR}/Submodules/AMReX/Tools/Plotfile/*/amrex_fextrema.exe CACHE INTERNAL "Path to fextrema executable for regression tests")
else()
    set(FCOMPARE_EXE ${CMAKE_BINARY_DIR}/Submodules/AMReX/Tools/Plotfile/amrex_fcompare CACHE INTERNAL "Path to fcompare executable for regression tests")
    set(FEXTREMA_EXE ${CMAKE_BINARY_DIR}/Submodules/AMReX/Tools/Plotfile/amrex_fextrema CACHE INTERNAL "Path to fextrema executable for regression tests")
endif()

# AMReX's particle_compare is a GNUmake-only tool, so build it here. Like
# fcompare it diffs plotfiles, but it compares particle data, which fcompare
# skips.
unset(PARTICLE_COMPARE_EXE)
if(REMORA_ENABLE_PARTICLES)
    set(_particle_compare_src "${PROJECT_SOURCE_DIR}/Submodules/AMReX/Tools/Postprocessing/C_Src/particle_compare.cpp")
    if(EXISTS "${_particle_compare_src}")
        add_executable(remora_particle_compare "${_particle_compare_src}")
        target_link_libraries(remora_particle_compare PRIVATE ${REMORA_AMREX_TARGET})
        if(REMORA_ENABLE_CUDA)
            setup_target_for_cuda_compilation(remora_particle_compare)
        endif()
        # CTestList uses the add_test(<name> <command>) form, which does not
        # expand generator expressions, so pin where the executable lands.
        set_target_properties(remora_particle_compare PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        set(PARTICLE_COMPARE_EXE "${CMAKE_CURRENT_BINARY_DIR}/remora_particle_compare${CMAKE_EXECUTABLE_SUFFIX}")
    else()
        message(WARNING "particle_compare.cpp not found in Submodules/AMReX; particle regression tests are disabled")
    endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/CTestList.cmake)

add_subdirectory(Unit)
