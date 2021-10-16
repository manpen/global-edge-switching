#pragma once

#include <cstddef>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
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
class AutocorrelationAnalysisOriginal {
public:
    AutocorrelationAnalysisOriginal() = delete;

    template <typename Gen, typename Graphseed, typename Seed>
    AutocorrelationAnalysisOriginal(const NetworKit::Graph& graph,
                            Gen& gen,
                            const std::vector<size_t>& thinnings,
                            size_t min_snapshots_per_thinning,
                            const std::string& algo_label,
                            const std::string& graph_label,
                            Graphseed& graphseed,
                            Seed& seed,
                            const std::string& output_fn_prefix,
                            int pu_id,
                            size_t switches_per_edge = 1,
                            size_t max_snapshots_per_thinning = std::numeric_limits<size_t>::max())
            : curr_graph(graph)
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
        // - per original edge a transition counter
        // - last considered snapshot
        std::vector<size_t> st_prev_snapshots(thinnings.size());
        std::vector<size_t> st_proc_snapshots(thinnings.size());
        std::vector<std::vector<transition_counter_t>> st_edge_transitions;
        std::vector<std::vector<bool>> st_edge_bits;
        st_edge_transitions.reserve(thinnings.size());
        st_edge_bits.reserve(thinnings.size());
        for (size_t tid = 0; tid < thinnings.size(); tid++) {
            st_edge_transitions.emplace_back(m);
            st_edge_bits.emplace_back(m, true);
        }
        std::vector<std::pair<NetworKit::node, NetworKit::node>> original_edges;
        original_edges.reserve(m);
        graph.forEdges([&](NetworKit::node u, NetworKit:: node v) {
            const auto ou = std::min(u, v);
            const auto ov = std::max(u, v);
            original_edges.emplace_back(ou, ov);
        });
        std::sort(original_edges.begin(), original_edges.end());



        std::cout
                << "# processing:\n"
                << "# algo " << algo_label << "\n"
                << "# graph " << graph_label << "\n"
                << "# n " << n << "\n"
                << "# true n " << true_n << "\n"
                << "# m " << m << "\n"
                << "# chain length " << min_chain_length << "\n"
                << "# individual snapshots " << snapshots.size() << "\n"
                << "# datastructure size in Bytes  " << sizeof(thinning_counter_t) * thinnings.size() * m << "\n"
                << "# has [node 0: " << (graph.hasNode(0) ? "y]" : "n]") << "\n"
                << "# has [node n: " << (graph.hasNode(n) ? "y]" : "n]")
                << std::endl;

        // perform the switchings and add edges to time series
        std::cout << "# performing switches" << std::endl;
        std::vector<size_t> successful_switches(s);
        size_t last_snapshot = 0;
        size_t last_reported_snapshot = 0;
        for (size_t snapshotid = 0; snapshotid < s; snapshotid++) {
            const auto snapshot = snapshots[snapshotid];
            if (last_reported_snapshot + min_chain_length / 10 < snapshot) {
                std::cout << "# snapshot " << snapshot << " / " << min_chain_length << " for " << output_fn_prefix << std::endl;
                last_reported_snapshot = snapshot;
            }

            // compute requested number of switches, filling the gap from last snapshot to this snapshot
            const auto factor = snapshot - last_snapshot;
            const auto requested_switches = factor * switches_per_edge * m/2 + 1;

            // perform switchings
            Algo es(curr_graph);
            successful_switches[snapshotid] = es.do_switches(gen, requested_switches);
            const auto& edgelist = es.get_edgelist();
            curr_graph = es.get_graph();

            // sort edgelist
            std::vector<std::pair<NetworKit::node, NetworKit::node>> pairs_edgelist;
            pairs_edgelist.reserve(m);
            for (const auto &edge : edgelist) {
                auto [u, v] = es::to_nodes(edge);
                const auto cu = std::min(u, v);
                const auto cv = std::max(u, v);
                pairs_edgelist.emplace_back(u, v);
            }
            std::sort(pairs_edgelist.begin(), pairs_edgelist.end());

            // compute indices of relevant thinnings
            std::vector<size_t> relevant_thinning_ids;
            relevant_thinning_ids.reserve(thinnings.size());
            for (size_t tid = 0; tid < thinnings.size(); tid++) {
                const auto prev_snapshot = st_prev_snapshots[tid];
                const auto thinning = thinnings[tid];
                const auto proc_snapshots = st_proc_snapshots[tid];
                if ((prev_snapshot + thinning == snapshot) && (proc_snapshots < max_snapshots_per_thinning)) {
                    relevant_thinning_ids.push_back(tid);
                }
            }

            auto update_edge = [&](size_t eid, bool next) {
                for (const auto tid : relevant_thinning_ids) {
                    // retrieve vectors
                    const auto &edge_bits = st_edge_bits[tid];
                    auto &edge_transitions = st_edge_transitions[tid];

                    // retrieve last edge existence bit and update
                    const auto prev = edge_bits[eid];
                    edge_transitions[eid].update(prev, next);
                }
            };

            std::vector<bool> edgelist_bits(m);
            auto eit_curr = pairs_edgelist.begin();
            size_t eid = 0;
            for (; eid < m;) {
                if (eit_curr == pairs_edgelist.end())
                    break;
                const auto oe = original_edges[eid];

                // move edge of current graph
                if (oe > *eit_curr) {
                    ++eit_curr;
                    continue;
                }
                // original edge does not exist in current graph
                if (oe < *eit_curr) {
                    update_edge(eid, false);
                    edgelist_bits[eid] = false;
                    eid++;
                    continue;
                }
                // original edge exists in current graph
                if (oe == *eit_curr) {
                    update_edge(eid, true);
                    edgelist_bits[eid] = true;
                    ++eit_curr;
                    eid++;
                }
            }
            for (size_t tail = eid; tail < m; ++tail) {
                 update_edge(eid, false);
                 edgelist_bits[eid] = false;
            }

            // update data structures
            for (const auto tid : relevant_thinning_ids) {
                // update for this thinning last considered snapshot and edgebits
                st_prev_snapshots[tid] = snapshot;
                st_proc_snapshots[tid]++;
                st_edge_bits[tid] = edgelist_bits;
            }

            last_snapshot = snapshot;
        }

        std::cout << "# done switching " << output_fn_prefix << std::endl;

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

        // open file to write to
        std::ofstream out_file(output_fn_prefix + std::to_string(pu_id) + ".log");
        for (size_t tid = 0; tid < thinnings.size(); tid++) {
            size_t thinning_successful_switches = 0;
            for (size_t sid = 0; sid <= thinning_snapshots[tid].back(); sid++)
                thinning_successful_switches += successful_switches[sid];

            thinning_counter_t eval_orig_sparse;
            for (const auto &orig_sparse_edge_transition : st_edge_transitions[tid]) {
                const double orig_sparse_delta_BIC = orig_sparse_edge_transition.compute_delta_BIC();
                eval_orig_sparse.update(false, orig_sparse_delta_BIC);
            }
            out_file  << "AUTOCORR,"
                      << algo_label << ","
                      << graph_label << ","
                      << true_n << ","
                      << m << ","
                      << min_chain_length << ","
                      << min_snapshots_per_thinning << ","
                      << max_snapshots_per_thinning << ","
                      << switches_per_edge << ","
                      << thinnings[tid] << ","
                      << st_proc_snapshots[tid] << ","
                      << thinning_successful_switches << ","
                      << 0 << ","
                      << 0 << ","
                      << eval_orig_sparse.num_independent << ","
                      << eval_orig_sparse.num_non_independent << ","
                      << (true_n)*(true_n - 1)/ 2 - eval_orig_sparse.num_independent - eval_orig_sparse.num_non_independent << ","
                      << graphseed << ","
                      << seed << "\n";
        }

        out_file.close();
    }

private:
    NetworKit::Graph curr_graph;
};