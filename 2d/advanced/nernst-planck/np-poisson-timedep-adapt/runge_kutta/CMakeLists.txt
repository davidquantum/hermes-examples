project(np-poisson-timedep-adapt-rk)

add_executable(${PROJECT_NAME} main.cpp ../definitions.cpp ../initial_conditions.cpp)

set_common_target_properties(${PROJECT_NAME} "HERMES2D")

