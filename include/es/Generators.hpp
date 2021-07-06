#pragma once

#include <random>
#include <es/Graph.hpp>

namespace es {

Graph generate_gnp(node_t n, double p, std::mt19937_64 &gen);

}