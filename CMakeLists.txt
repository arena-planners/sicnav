cmake_minimum_required(VERSION 3.12)
project(arena_planners_sicnav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(
  PROJECTFILE pyproject.toml
  POST_INSTALL "--no-build-isolation git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d"
)

install(
  FILES
    planner.py
    planner.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
