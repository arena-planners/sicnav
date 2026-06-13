cmake_minimum_required(VERSION 3.12)
project(arena_planners_sicnav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(PROJECTFILE pyproject.toml)

# Python-RVO2 (Cython, no PyPI wheel) is imported at module load by
# sicnav.utils.mpc_utils.orca_c_wrapper. Build it into the venv after creation.
# Pinned commit avoids upstream main drift; --no-build-isolation reuses the
# venv's cython==0.29.33 to keep the ABI consistent.
add_custom_command(
  OUTPUT "${venv_dir}/.rvo2_built"
  DEPENDS "${venv_dir}/.built"
  COMMAND sh -c "export PYTHONPATH='' && . '${venv_dir}/bin/activate' && uv pip install --no-build-isolation 'git+https://github.com/sybrenstuvel/Python-RVO2.git@b272b378e0a3bd4a6f1c7be6531a2fa14a7d5c80'"
  COMMAND ${CMAKE_COMMAND} -E touch "${venv_dir}/.rvo2_built"
  VERBATIM
)
add_custom_target(build_python_rvo2 ALL
  DEPENDS "${venv_dir}/.rvo2_built"
)
add_dependencies(build_python_rvo2 "${target_venv}")

install(
  FILES
    planner.py
    planner.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
