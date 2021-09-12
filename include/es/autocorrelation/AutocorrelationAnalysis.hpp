#pragma once
#ifndef EDGE_SWITCHING_AUTOCORRELATION_HPP
#define EDGE_SWITCHING_AUTOCORRELATION_HPP

#include <cstddef>
#include <vector>
#include <set>
#include <iostream>
#include <random>
#include <tsl/robin_set.h>
#include <es/Graph.hpp>
#include <tlx/unused.hpp>

size_t ggt(size_t a, size_t b) {
    while (b != 0) {
        size_t h = a % b;
        a = b;
        b = h;
    }
    return a;
}

size_t kgv(size_t a, size_t b) {
    if (a == 1 || b == 1) return a * b;
    return a * b / ggt(a, b);
}

size_t kgv(const std::vector<size_t>& v) {
    if (v.size() == 1)
        return v.front();
    else {
        size_t t = 1;
        for (const auto u : v) t = kgv(t, u);
        return t;
    }
}

struct thinning_counter_t {
    size_t num_none_independent = 0;
    size_t num_non_independent = 0;
    size_t num_independent = 0;
    thinning_counter_t() = default;

    void update(bool none, double delta_BIC) {
        num_none_independent += (none && delta_BIC < 0);
        num_independent += (!none && delta_BIC < 0);
        num_non_independent += (!none && delta_BIC >= 0);
    }
};

struct transition_counter_t {
    double x00;
    double x01;
    double x10;
    double x11;
    transition_counter_t(double y00, double y01, double y10, double y11) : x00(y00), x01(y01), x10(y10), x11(y11) { }

    void update(bool p, bool n) {
        x00 = x00 + (!p && !n);
        x01 = x01 + (!p && n);
        x10 = x10 + (p && !n);
        x11 = x11 + (p && n);
    }

    [[nodiscard]] double sum() const {
        return x00 + x01 + x10 + x11;
    }

    [[nodiscard]] transition_counter_t compute_independent_model_prediction() const {
        const double sum_x = sum();
        return transition_counter_t{
                (x00 + x01) * (x00 + x10) / sum_x,
                (x00 + x01) * (x01 + x11) / sum_x,
                (x10 + x11) * (x00 + x10) / sum_x,
                (x10 + x11) * (x01 + x11) / sum_x
        };
    }

    [[nodiscard]] double compute_delta_BIC() const {
        const transition_counter_t x_I_hat = compute_independent_model_prediction();
        const double x00G2 = (x00 == 0. ? 0. : x00*log(x_I_hat.x00/x00));
        const double x01G2 = (x01 == 0. ? 0. : x01*log(x_I_hat.x01/x01));
        const double x10G2 = (x10 == 0. ? 0. : x10*log(x_I_hat.x10/x10));
        const double x11G2 = (x11 == 0. ? 0. : x11*log(x_I_hat.x11/x11));
        const double delta_BIC = (-2.) * (x00G2 + x01G2 + x10G2 + x11G2) - log(sum());
        return delta_BIC;
    }

    [[nodiscard]] bool is_none() const {
        return x00 == sum();
    }
};



template <typename Algo>
class TimeSeries {
    using edge_series_t = tsl::robin_set<es::edge_t, es::edge_hash_crc32>;

public:
    template <typename Gen, typename Graphseed, typename Seed>
    TimeSeries(const NetworKit::Graph& graph,
               Gen& gen,
               const std::vector<size_t>& thinnings,
               size_t min_snapshots,
               const std::string& algo_label,
               const std::string& graph_label,
               Graphseed& graphseed,
               Seed& seed,
               size_t switches_per_edge = 1,
               size_t max_snapshots_per_thinning = std::numeric_limits<size_t>::max())
            : curr_graph(graph),
              m_num_nodes(graph.numberOfNodes())
    {
        const auto thinnings_kgv = kgv(thinnings);
        const auto min_chain_length = std::max(thinnings_kgv, min_snapshots / (thinnings_kgv / thinnings.back()) * thinnings_kgv);
        std::set<size_t> snapshots_set;
        for (const auto thinning : thinnings) {
            for (size_t i = 0; (i < min_chain_length / thinning) && (i < max_snapshots_per_thinning); i++) {
                snapshots_set.insert((i + 1) * thinning);
            }
        }
        const auto n = graph.numberOfNodes();
        const auto m = graph.numberOfEdges();
        const auto s = snapshots_set.size();
        std::vector<size_t> snapshots;
        std::copy(snapshots_set.begin(), snapshots_set.end(), std::back_inserter(snapshots));
        m_snapshots_edges.resize(s*(n + 1)*n/2);

        std::cout
                << "# processing:\n"
                << "# algo " << algo_label << "\n"
                << "# graph " << graph_label << "\n"
                << "# n " << n << "\n"
                << "# m " << m << "\n"
                << "# chain length " << min_chain_length << "\n"
                << "# individual snapshots " << s << "\n"
                << "# number of snapshot edge bits " << m_snapshots_edges.size() << "\n"
                << "# has [node 0: " << (graph.hasNode(0) ? "y]" : "n]") << "\n"
                << "# has [node n: " << (graph.hasNode(n) ? "y]" : "n]")
                << std::endl;

        // perform the switchings and add edges to time series
        std::cout << "# performing switches" << std::endl;
        size_t last_snapshot = 0;
        for (size_t snapshotid = 0; snapshotid < s; snapshotid++) {
            const auto snapshot = snapshots[snapshotid];

            // compute requested number of switches, filling the gap from last snapshot to this snapshot
            const auto factor = snapshot - last_snapshot;
            const auto requested_switches = factor * switches_per_edge * graph.numberOfEdges();

            Algo es(curr_graph);
            const auto successful_switches = es.do_switches(gen, requested_switches);
            const auto &edgelist = es.get_edgelist();

            for (size_t eid = 0; eid < edgelist.size(); eid++) {
                const auto e = edgelist[eid];
                const auto [u, v] = es::to_nodes(e);
                const auto ts_index = get_index(e);
                assert(s * ts_index + snapshotid < m_snapshots_edges.size());
                m_snapshots_edges[s * ts_index + snapshotid] = true;
            }

            last_snapshot = snapshot;
            curr_graph = es.get_graph();
        }

        // compute snapshots to consider per thinning
        std::vector<std::vector<size_t>> thinning_snapshots(thinnings.size());
        for (size_t thinningid = 0; thinningid < thinnings.size(); thinningid++) {
            const auto thinning = thinnings[thinningid];
            for (size_t sid = 0; sid < snapshots.size() && (thinning_snapshots[thinningid].size() < max_snapshots_per_thinning); sid++) {
                const auto snapshot = snapshots[sid];
                if (snapshot == (thinning_snapshots[thinningid].size() + 1) * thinning) {
                    thinning_snapshots[thinningid].push_back(sid);
                }
            }
        }

        // compute independence rates
        const auto pus = omp_get_max_threads();
        omp_set_num_threads(pus);
        std::vector<std::vector<thinning_counter_t>> thinning_counters(pus);
        for (auto &counter : thinning_counters) {
            for (const auto thinning : thinnings) {
                counter.emplace_back();
            }
        }
        std::cout << "# processing time series [pus = " << pus << "]" << std::endl;
#pragma omp parallel
        {
            const auto pu = omp_get_thread_num();

            // compute begin and end for this pu
            const auto fbeg = pu * m_snapshots_edges.size() / pus;
            const auto fend = (pu + 1) * m_snapshots_edges.size() / pus;
            const auto beg = (fbeg % s != 0 ? s * (fbeg / s + 1) : fbeg);
            const auto end = (fend % s != 0 ? s * (fend / s + 1) : fend);
            assert((pu < pus) || (fend == m_snapshots_edges.size()));
            assert((pu < pus) || (end == m_snapshots_edges.size()));
            assert((end - beg) % s == 0);

            es::edge_t curr_edge = get_edge(beg / s);
            assert(get_index(curr_edge) == beg / s);

            for (size_t edgeblock = 0; edgeblock < (end - beg) / s; edgeblock++) {
                const auto eb_beg = beg + edgeblock * s;
                const auto eb_end = beg + (edgeblock + 1) * s;
                tlx::unused(eb_end);
                const auto [u, v] = es::to_nodes(curr_edge);
                for (size_t thinningid = 0; thinningid < thinnings.size(); thinningid++) {
                    const auto thinning = thinnings[thinningid];
                    bool edge_exists = graph.hasEdge(u, v) || graph.hasEdge(v, u);
                    transition_counter_t edge_transitions{0., 0., 0., 0.};

                    for (const auto sid : thinning_snapshots[thinningid]) {
                        const auto snapshot = snapshots[sid];
                        assert(eb_beg + sid < eb_end);
                        edge_transitions.update(edge_exists, m_snapshots_edges[eb_beg + sid]);
                        edge_exists = m_snapshots_edges[eb_beg + sid];
                    }

                    const double delta_BIC = edge_transitions.compute_delta_BIC();
                    thinning_counters[pu][thinningid].update(edge_transitions.is_none(), delta_BIC);
                }

                curr_edge = get_next_edge(curr_edge);
            }
        }

        // compute number of considered snapshots per thinning
        std::vector<size_t> snapshots_per_thinning;
        for (const auto thinning : thinnings) {
            size_t considered_snapshots = 0;
            for (size_t sid = 0; (sid < snapshots.size()) && (considered_snapshots < max_snapshots_per_thinning); sid++) {
                const auto snapshot = snapshots[sid];
                if (snapshot == (considered_snapshots + 1) * thinning) considered_snapshots++;
            }
            snapshots_per_thinning.emplace_back(considered_snapshots);
        }

        // combine parallely computed number of independent edges for each thinning parameter
        std::cout << "type,algo,graphlabel,n,m,chainlength,max snapshots/thinning,switches/edge,thinning,snapshots/thinning,independent edges,non-independent edges,independent none-edges,non-independent none-edges,graphseed,seed" << std::endl;
        std::vector<thinning_counter_t> final_counters;
        for (size_t i = 0; i < thinnings.size(); i++) {
            thinning_counter_t t;
            for (int j = 0; j < omp_get_max_threads(); j++) {
                t.num_none_independent += thinning_counters[j][i].num_none_independent;
                t.num_independent += thinning_counters[j][i].num_independent;
                t.num_non_independent += thinning_counters[j][i].num_non_independent;
            }
            t.num_none_independent -= (graph.hasNode(0)) * (n - 1);
            t.num_none_independent -= (graph.hasNode(n)) * (n - 1);

            std::cout << "AUTOCORR,"
            << algo_label << ","
            << graph_label << ","
            << n << ","
            << m << ","
            << min_chain_length << ","
            << max_snapshots_per_thinning << ","
            << switches_per_edge << ","
            << thinnings[i] << ","
            << snapshots_per_thinning[i] << ","
            << t.num_independent << ","
            << t.num_non_independent << ","
            << t.num_none_independent << ","
            << graphseed << ","
            << seed << std::endl;
        }
    }

private:
    const es::node_t m_num_nodes;
    std::vector<bool> m_snapshots_edges;
    NetworKit::Graph curr_graph;

    size_t get_index(es::edge_t e) {
        const auto uv = es::to_nodes(e);
        const auto u = uv.first;
        const auto v = uv.second;
        const auto d = m_num_nodes + 1;
        return (d*(d-1)/2) - (d-u) * ((d-u) - 1)/2 + v - u - 1;
    }

    es::edge_t get_edge(size_t i) {
        const auto d = m_num_nodes + 1;
        const auto u = d - 2 - static_cast<es::node_t>(std::floor(std::sqrt(-8*i + 4*d*(d-1) - 7)/2. - 0.5));
        const auto v = i + u + 1 - d*(d-1)/2 + (d-u)*((d-u)-1)/2;
        return es::to_edge(u, v);
    }

    es::edge_t get_next_edge(es::edge_t e) {
        const auto uv = es::to_nodes(e);
        const auto u = uv.first;
        const auto v = uv.second;
        if (v == m_num_nodes) {
            return es::to_edge(u + 1, u + 2);
        } else {
            return es::to_edge(u, v + 1);
        }
    }
};

#endif //EDGE_SWITCHING_AUTOCORRELATION_HPP