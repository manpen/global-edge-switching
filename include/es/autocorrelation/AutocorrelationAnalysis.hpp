#pragma once
#ifndef EDGE_SWITCHING_AUTOCORRELATION_HPP
#define EDGE_SWITCHING_AUTOCORRELATION_HPP

#include <cstddef>
#include <vector>
#include <set>
#include <iostream>
#include <random>
#include <es/Graph.hpp>

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
    size_t num_non_independent = 0;
    size_t num_independent = 0;
    thinning_counter_t() = default;

    void update(bool none, double delta_BIC) {
        num_independent += (!none && delta_BIC < 0);
        num_non_independent += (!none && delta_BIC >= 0);
    }
};

struct transition_counter_t {
    double x00;
    double x01;
    double x10;
    double x11;
    transition_counter_t() = default;
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
class AutocorrelationAnalysis {
public:
    AutocorrelationAnalysis() = delete;

    template <typename Gen, typename Graphseed, typename Seed>
    AutocorrelationAnalysis(const NetworKit::Graph& graph,
                            Gen& gen,
                            const std::vector<size_t>& thinnings,
                            size_t min_snapshots_per_thinning,
                            const std::string& algo_label,
                            const std::string& graph_label,
                            Graphseed& graphseed,
                            Seed& seed,
                            size_t switches_per_edge = 1,
                            size_t max_snapshots_per_thinning = std::numeric_limits<size_t>::max(),
                            int pus = 1)
            : curr_graph(graph),
              m_num_nodes(graph.numberOfNodes())
    {
        const auto thinnings_kgv = kgv(thinnings);
        const auto min_snapshots_filler = min_snapshots_per_thinning / (thinnings_kgv / thinnings.back())
                                          + (min_snapshots_per_thinning % (thinnings_kgv / thinnings.back()) != 0);
        const auto min_chain_length = std::max(thinnings_kgv, min_snapshots_filler * thinnings_kgv);
        std::set<size_t> snapshots_set;
        for (const auto thinning : thinnings) {
            for (size_t i = 0; (i < min_chain_length / thinning) && (i < max_snapshots_per_thinning); i++) {
                snapshots_set.insert((i + 1) * thinning);
            }
        }
        const auto n = graph.numberOfNodes();
        const auto m = graph.numberOfEdges();
        const auto s = snapshots_set.size();
        NetworKit::node true_n = 0;
        for (NetworKit::node node = 0; node < n; node++) true_n += (graph.degree(node) > 0);
        std::vector<size_t> snapshots;
        std::copy(snapshots_set.begin(), snapshots_set.end(), std::back_inserter(snapshots));

        // data structure for the thinnings contains
        // - per possible edge a transition counter
        // - last considered snapshot
        const size_t num_possible_edges = n * (n - 1)/2;
        std::vector<size_t> t_prev_snapshots(thinnings.size());
        std::vector<size_t> t_proc_snapshots(thinnings.size());
        std::vector<std::vector<transition_counter_t>> t_edge_transitions;
        std::vector<std::vector<bool>> t_edge_bits;
        t_edge_transitions.reserve(thinnings.size());
        t_edge_bits.reserve(thinnings.size());
        for (size_t tid = 0; tid < thinnings.size(); tid++) {
            t_edge_transitions.emplace_back(num_possible_edges);
            t_edge_bits.emplace_back(num_possible_edges);
        }

        std::vector<size_t> orig_bit_indices;
        orig_bit_indices.reserve(graph.numberOfEdges());
        graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
            orig_bit_indices.push_back(get_index(es::to_edge(u, v)));
            for (auto &edge_bits : t_edge_bits) {
                assert(get_index(es::to_edge(u, v)) < t_edge_bits[0].size());
                edge_bits[get_index(es::to_edge(u, v))] = true;
            }
        });

        std::cout
                << "# processing:\n"
                << "# algo " << algo_label << "\n"
                << "# graph " << graph_label << "\n"
                << "# n " << n << "\n"
                << "# true n " << true_n << "\n"
                << "# m " << m << "\n"
                << "# chain length " << min_chain_length << "\n"
                << "# individual snapshots " << snapshots.size() << "\n"
                << "# datastructure size in Bytes  " << sizeof(thinning_counter_t) * thinnings.size() * num_possible_edges << "\n"
                << "# has [node 0: " << (graph.hasNode(0) ? "y]" : "n]") << "\n"
                << "# has [node n: " << (graph.hasNode(n) ? "y]" : "n]")
                << std::endl;

        // perform the switchings and add edges to time series
        std::cout << "# performing switches" << std::endl;
        std::vector<size_t> successful_switches(s);
        size_t last_snapshot = 0;
        for (size_t snapshotid = 0; snapshotid < s; snapshotid++) {
            const auto snapshot = snapshots[snapshotid];

            // compute requested number of switches, filling the gap from last snapshot to this snapshot
            const auto factor = snapshot - last_snapshot;
            const auto requested_switches = factor * switches_per_edge * graph.numberOfEdges() / 2 + 1;

            // perform switchings
            Algo es(curr_graph);
            successful_switches[snapshotid] = es.do_switches(gen, requested_switches, true);
            const auto &edgelist = es.get_edgelist();

            // copy edgelist to bit representation
            std::vector<bool> sw_edge_bits(num_possible_edges);
            for (const auto &edge : edgelist) {
                sw_edge_bits[get_index(edge)] = true;
            }

            // iterate over different thinnings and perform updates
            for (size_t tid = 0; tid < thinnings.size(); tid++) {
                const auto prev_snapshot = t_prev_snapshots[tid];
                const auto thinning = thinnings[tid];
                if (prev_snapshot + thinning == snapshot) {
                    const auto &edge_bits = t_edge_bits[tid];
                    auto &edge_transitions = t_edge_transitions[tid];
                    assert(edge_bits.size() == sw_edge_bits.size());
                    for (size_t bitid = 0; bitid < edge_bits.size(); bitid++) {
                        const auto prev = edge_bits[bitid];
                        const auto next = sw_edge_bits[bitid];
                        edge_transitions[bitid].update(prev, next);
                    }

                    // update for this thinning last considered snapshot and edgebits
                    t_prev_snapshots[tid] = snapshot;
                    t_edge_bits[tid] = sw_edge_bits;
                    t_proc_snapshots[tid]++;
                }
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

        const es::edge_hash_crc32 h;
        for (size_t tid = 0; tid < thinnings.size(); tid++) {
            size_t thinning_successful_switches = 0;
            for (size_t sid = 0; sid <= thinning_snapshots[tid].back(); sid++)
                thinning_successful_switches += successful_switches[sid];

            thinning_counter_t eval_all;
            thinning_counter_t eval_orig;
            const auto &edge_transitions = t_edge_transitions[tid];
            es::edge_t curr_edge = get_edge(0);
            for (const auto &edge_transition : edge_transitions) {
                const double delta_BIC = edge_transition.compute_delta_BIC();
                eval_all.update(edge_transition.is_none(), delta_BIC);

                curr_edge = get_next_edge(curr_edge);
            }
            for (const auto &orig_bit_index : orig_bit_indices) {
                const auto orig_edge_transition = edge_transitions[orig_bit_index];
                const double orig_delta_BIC = orig_edge_transition.compute_delta_BIC();
                eval_orig.update(orig_edge_transition.is_none(), orig_delta_BIC);
            }

            std::cout << "AUTOCORR,"
                      << algo_label << ","
                      << graph_label << ","
                      << true_n << ","
                      << m << ","
                      << min_chain_length << ","
                      << min_snapshots_per_thinning << ","
                      << max_snapshots_per_thinning << ","
                      << switches_per_edge << ","
                      << thinnings[tid] << ","
                      << t_proc_snapshots[tid] << ","
                      << thinning_successful_switches << ","
                      << eval_all.num_independent << ","
                      << eval_all.num_non_independent << ","
                      << eval_orig.num_independent << ","
                      << eval_orig.num_non_independent << ","
                      << (true_n)*(true_n - 1)/ 2 - eval_all.num_independent - eval_all.num_non_independent << ","
                      << graphseed << ","
                      << seed << std::endl;
        }
    }

private:
    const es::node_t m_num_nodes;
    NetworKit::Graph curr_graph;

    size_t get_index(es::edge_t e) {
        const auto uv = es::to_nodes(e);
        const auto u = uv.first;
        const auto v = uv.second;
        const auto d = m_num_nodes;
        return (d*(d-1)/2) - (d-u) * ((d-u) - 1)/2 + v - u - 1;
    }

    es::edge_t get_edge(size_t i) {
        const auto d = m_num_nodes;
        const auto u = d - 2 - static_cast<es::node_t>(std::floor(std::sqrt(-8*i + 4*d*(d-1) - 7)/2. - 0.5));
        const auto v = i + u + 1 - d*(d-1)/2 + (d-u)*((d-u)-1)/2;
        return es::to_edge(u, v);
    }

    es::edge_t get_next_edge(es::edge_t e) {
        const auto uv = es::to_nodes(e);
        const auto u = uv.first;
        const auto v = uv.second;
        if (v == m_num_nodes - 1) {
            return es::to_edge(u + 1, u + 2);
        } else {
            return es::to_edge(u, v + 1);
        }
    }
};

#endif //EDGE_SWITCHING_AUTOCORRELATION_HPP
