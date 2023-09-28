/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#include "xcl2.hpp"
#include <unistd.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <bitset>
#include <string>
#include <random>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <fstream>

//#define DATA_SIZE 256
//#define DATA_SIZE 1024        // 64 transactions
//#define DATA_SIZE 4096        // 256 transactions
//#define DATA_SIZE 4096        // 256 transactions
//#define DATA_SIZE 17600       // 1100 transactions
//#define DATA_SIZE 176000       // 11000 transactions
//#define DATA_SIZE 4194304       // 4 Mi ints -> 16 MiB
#define DATA_SIZE 16777216    // 16 Mi ints -> 64 MiB
//#define DATA_SIZE 167772160    // 160 Mi ints -> 640 MiB
//#define DATA_SIZE 1677721600    // 1600 Mi ints -> 6400 MiB
//#define DATA_SIZE 268435456   // 256 Mi ints ->  1 GiB

#define TX_SIZE 512
#define TREE_ID_NUM_BITS 22
#define CLASS_ID_NUM_BITS 10
#define MACRO_WORD_SIZE 32
//#define WORD_SIZE 32
//#define WORD_SIZE 16
#define WORD_SIZE 8
//#define WORD_SIZE 4

using namespace std;

void nice_print_vector(vector<int, aligned_allocator<int>> &vec, unsigned long useful_size) {
    for (unsigned long i = 0; i < useful_size; i++) {
        if (i % 16 == 0 && i > 0) {
            cout << endl;
        }
        cout << "0x" << std::setfill ('0') << std::setw(sizeof(unsigned)*2) << hex << vec[i] << ", ";
    }
    cout << endl;
}

unsigned **compact_id;

bool should_negate(unsigned num_branches, unsigned branch_id, unsigned var_id) {
    // TODO: add assert to ensure num_branches is an exact power of two
    unsigned num_vars = (unsigned) lround(log2(num_branches));

    return (branch_id >> (num_vars - (var_id + 1))) % 2;    
}

void init_compact_id(unsigned num_branches) {
    // TODO: add assert to ensure num_branches is an exact power of two
    unsigned num_vars = (unsigned) lround(log2(num_branches));

    compact_id = (unsigned **) malloc(num_branches * sizeof(unsigned *));
    for (unsigned i = 0; i < num_branches; i++) {
        compact_id[i] = (unsigned *) malloc(num_vars * sizeof(unsigned));

        for (unsigned j = 0; j < num_vars; j++) {
            //compact_id[i][j] = (unsigned) ((i >> (num_vars - j)) + (1 << lround(floor(log2(j + 1)))) - 1);
            //compact_id[i][j] = (unsigned) ((i >> (num_vars - j)) + (1 << lround(log2(j + 1))) - 1);
            compact_id[i][j] = (unsigned) ((i >> (num_vars - j)) + (1 << j) - 1);
            cout << (should_negate(num_branches, i, j) ? "~" : " ");
            cout << compact_id[i][j] << ", ";
        }

        cout << endl;
    }
}

void print_csv_vector(vector<vector<int>> csv_vector) {
    for (const auto& row: csv_vector) {
        for (const auto& field: row) {
            cout << field << "\t";
        }
        cout << endl;
    }
}

vector<vector<int>> read_csv(string basePath, string filePath) {
    ifstream file(basePath + "/" + filePath);

    if(!file.is_open()) {
        cerr << "[test_csv_parsing]: File is unavailable." << endl;
        exit(1);
    }

    vector<vector<int>> rows;

    string line;
    while(getline(file, line)) {
        istringstream sline(line);
        vector<int> fields;
        string field;

        while(getline(sline, field, ',')) {
            fields.push_back(stoi(field));
        }

        rows.push_back(fields);
    }

    file.close();

    return rows;
}

class BranchPair {
public:
    unsigned low;
    unsigned high;
    bool negate;

    BranchPair(): low(0), high(0), negate(false) {}

    BranchPair(unsigned low_ex, unsigned high_ex): low(low_ex), high(high_ex), negate(false) {}

    BranchPair(unsigned low_ex, unsigned high_ex, bool negate): low(low_ex), high(high_ex), negate(negate) {}
};

ostream& operator<< (ostream& os, const BranchPair& pair) {
    os << "(" << pair.low << ", " << pair.high << ")";
    return os;
}

// Notice that we use the move operator
// whenever possible to avoid copies.
// As a result, passed data might cease
// to be available from call site.
template <unsigned num_branches, unsigned max_num_vars, unsigned max_state> class Input {
public:
    unsigned *queries;
    unsigned num_queries;
    unsigned num_vars;
private:
    //vector<vector<BranchPair>> branches;
    BranchPair *branches;
    int *leaves;
    unsigned *tree_ids;
    unsigned *class_ids;
    vector<unsigned> dont_cares;
    bitset<max_num_vars> dont_cares_set;
    const unsigned min_pair_length;
    unsigned num_cores;
    unsigned num_trees;
    const unsigned min_branches_per_match;
    bool enable_leaf_values;
    bool enable_compact_trees;
    
    random_device rd;
    //minstd_rand gen;
    mt19937 gen;

    // The following two elements are only used when
    // data is loaded from CSV.
    bool loaded_from_file = false;
    unsigned num_branches_with_padding;

    unsigned rand_u(unsigned low, unsigned high)
    {
        uniform_int_distribution<unsigned> distribution(low, high);

        return distribution(gen);
    }

public:

    void update_thresholds() {
        #pragma omp parallel
        {
            unsigned b_id, v_id, t_low, t_high, c_id;
            random_device p_rd;
            //minstd_rand p_gen;
            mt19937 p_gen;
            p_gen.seed(p_rd());

            uniform_int_distribution<unsigned> low_t_dist(0, max_state - min_pair_length);
            uniform_int_distribution<int> leaf_value_dist(-255, 255);
            uniform_int_distribution<unsigned> tree_id_dist(1, 255);
            uniform_int_distribution<unsigned> class_id_dist(1, 16);

            if (enable_leaf_values) {
                #pragma omp for
                for (b_id = 0; b_id < num_branches * num_cores; b_id++) {
                    for (v_id = 0; v_id < num_vars; v_id++) {
                        t_low = low_t_dist(p_gen);

                        uniform_int_distribution<unsigned> high_t_dist(t_low + min_pair_length, max_state);
                        t_high = high_t_dist(p_gen);
                        
                        branches[b_id * num_vars + v_id] = BranchPair(t_low, t_high);
                    }

                    leaves[b_id] = leaf_value_dist(p_gen);
                    tree_ids[b_id] = tree_id_dist(p_gen);
                    class_ids[b_id] = class_id_dist(p_gen);
                }
                
#ifdef VERBOSE
                #pragma omp single
                for (b_id = 0; b_id < num_branches; b_id++) {
                    cout << "(branch_id, leaf_value, tree_id, class_id) = (" << b_id << ", " << hex << leaves[b_id] << ", " << tree_ids[b_id] << ", " << class_ids[b_id] << ")" << endl;
                }
#endif
            } else {
                #pragma omp for
                for (b_id = 0; b_id < num_branches; b_id++) {
                    for (v_id = 0; v_id < num_vars; v_id++) {
                        t_low = low_t_dist(p_gen);

                        uniform_int_distribution<unsigned> high_t_dist(t_low + min_pair_length, max_state);
                        t_high = high_t_dist(p_gen);
                        
                        branches[b_id * num_vars + v_id] = BranchPair(t_low, t_high);
                    }
                }
            }
        }
    }

    Input(unsigned num_dont_cares, unsigned num_queries_c, double min_var_match_prob, bool enable_leaf_values = true, unsigned min_branches_per_match = 64, unsigned num_cores = 1, bool enable_compact_trees = false, unsigned num_vars = max_num_vars) :
    min_pair_length(min(unsigned(lround(ceil(min_var_match_prob * (max_state + 1)))), max_state)),
    min_branches_per_match(min_branches_per_match),
    num_cores(num_cores),
    num_vars(num_vars),
    enable_leaf_values(enable_leaf_values),
    enable_compact_trees(enable_compact_trees){
        gen.seed(rd());

        assert(num_vars >= num_dont_cares);
        num_queries = num_queries_c;

        for (unsigned dc_id = 0; dc_id < num_dont_cares; dc_id++) {
            while (true) {
                unsigned v_id = rand_u(0, num_vars - 1);

                if (!dont_cares_set[v_id]) {
                    dont_cares.push_back(v_id);
                    dont_cares_set[v_id] = true;
                    break;
                }
            }
        }

        cout << "Just generated the don't cares" << endl;
        branches = (BranchPair *) malloc(sizeof(BranchPair) * num_branches * num_vars * num_cores);
        queries = (unsigned *) malloc(sizeof(unsigned) * num_queries * num_vars);
        leaves = (int *) malloc(sizeof(unsigned) * num_branches * num_cores);
        tree_ids = (unsigned *) malloc(sizeof(unsigned) * num_branches * num_cores);
        class_ids = (unsigned *) malloc(sizeof(unsigned) * num_branches * num_cores);

        #pragma omp parallel
        {
            unsigned b_id, v_id, t_low, t_high, c_id;
            random_device p_rd;
            //minstd_rand p_gen;
            mt19937 p_gen;
            p_gen.seed(p_rd());

            uniform_int_distribution<unsigned> low_t_dist(0, max_state - min_pair_length);
            uniform_int_distribution<int> leaf_value_dist(-255, 255);
            uniform_int_distribution<unsigned> tree_id_dist(1, 255);
            uniform_int_distribution<unsigned> class_id_dist(1, 16);

            if (enable_leaf_values) {
                #pragma omp for
                for (b_id = 0; b_id < num_branches * num_cores; b_id++) {
                    for (v_id = 0; v_id < num_vars; v_id++) {
                        t_low = low_t_dist(p_gen);

                        uniform_int_distribution<unsigned> high_t_dist(t_low + min_pair_length, max_state);
                        t_high = high_t_dist(p_gen);
                        
                        branches[b_id * num_vars + v_id] = BranchPair(t_low, t_high);
                    }

                    leaves[b_id] = leaf_value_dist(p_gen);
                    tree_ids[b_id] = tree_id_dist(p_gen);
                    class_ids[b_id] = class_id_dist(p_gen);
                }
                    
                #pragma omp single
                for (b_id = 0; b_id < num_branches; b_id++) {
                    cout << "(branch_id, leaf_value, tree_id, class_id) = (" << b_id << ", " << hex << leaves[b_id] << ", " << tree_ids[b_id] << ", " << class_ids[b_id] << ")" << endl;
                }
            } else {
                #pragma omp for
                for (b_id = 0; b_id < num_branches; b_id++) {
                    for (v_id = 0; v_id < num_vars; v_id++) {
                        t_low = low_t_dist(p_gen);

                        uniform_int_distribution<unsigned> high_t_dist(t_low + min_pair_length, max_state);
                        t_high = high_t_dist(p_gen);
                        
                        branches[b_id * num_vars + v_id] = BranchPair(t_low, t_high);
                    }
                }
            }

            #pragma omp single
            if (enable_leaf_values) {
                cout << "Just generated the branches and leaf values" << endl;
            } else {
                cout << "Just generated the branches" << endl;
            }

            unsigned q_id;
            uniform_int_distribution<unsigned> q_dist(0, max_state);

            #pragma omp for
            for (q_id = 0; q_id < num_queries; q_id++) {

                for (v_id = 0; v_id < num_vars; v_id++) {
                    queries[q_id * num_vars + v_id] = q_dist(p_gen);
                }
            }

            #pragma omp single
            cout << "Just generated the queries" << endl;
        }
    }

    Input(string basePath, bool enable_leaf_values = true, unsigned min_branches_per_match = 16, unsigned num_cores = 1, bool enable_compact_trees = false) :
    min_pair_length(0),
    min_branches_per_match(min_branches_per_match),
    num_cores(num_cores),
    enable_leaf_values(enable_leaf_values),
    enable_compact_trees(enable_compact_trees){
        vector<vector<int>> T_quant_csv = read_csv(basePath, "T_quant.csv");
        vector<vector<int>> L_quant_csv = read_csv(basePath, "L_quant.csv");
        vector<vector<int>> t_ids_csv = read_csv(basePath, "t_ids.csv");
        vector<vector<int>> c_ids_csv = read_csv(basePath, "c_ids.csv");
        vector<vector<int>> X_test_quant_csv = read_csv(basePath, "X_test_quant.csv");

        loaded_from_file = true;

        //cout << "[Input::csv_constructor]: T_quant" << endl;
        //print_csv_vector(T_quant_csv);

        num_trees = 0;
        unsigned total_valid_branches = 0;

        // TODO: Substitute the following with information obtained
        //       by a configuration file or dataset CSV
        unsigned num_branches_per_tree = min_branches_per_match;

        if (!t_ids_csv.empty()) {
            num_trees = t_ids_csv.back()[0] + 1;
            total_valid_branches = t_ids_csv.size();
            cout << "Num trees: " << num_trees << endl;
            cout << "Num valid branches: " << total_valid_branches << endl;
            num_vars = X_test_quant_csv[0].size();
        }

        assert(total_valid_branches > 0);

        num_queries = X_test_quant_csv.size();
        //num_vars = X_test_quant_csv[0].size();
        cout << "Num queries: " << num_queries << endl;
        cout << "Num vars: " << num_vars << endl;

        num_branches_with_padding = num_trees * num_branches_per_tree;

        branches = (BranchPair *) malloc(sizeof(BranchPair) * num_branches_with_padding * num_vars);
        queries = (unsigned *) malloc(sizeof(unsigned) * num_queries * num_vars);
        leaves = (int *) malloc(sizeof(unsigned) * num_branches_with_padding);
        tree_ids = (unsigned *) malloc(sizeof(unsigned) * num_branches_with_padding);
        class_ids = (unsigned *) malloc(sizeof(unsigned) * num_branches_with_padding);

        vector<unsigned> first_ids_by_branch;
        
        first_ids_by_branch.push_back(0);
        unsigned last_id = t_ids_csv[0][0];

        for (unsigned i = 1; i < total_valid_branches; i++) {
            unsigned current_id = t_ids_csv[i][0];        

            if (current_id != last_id) {
                first_ids_by_branch.push_back(i);
                last_id = current_id;
            }
        }

        // Initialize everything with zeros
        for (unsigned i = 0; i < num_branches_with_padding; i++) {
            for (unsigned v_id = 0; v_id < num_vars; v_id++) {
                BranchPair p;

                branches[i * num_vars + v_id] = p;
            }

            leaves[i] = 0;
            tree_ids[i] = 0;
            class_ids[i] = 0;
        }

        for (unsigned i = 0; i < total_valid_branches; i++) {
            unsigned tree_id = t_ids_csv[i][0];
            unsigned class_id = c_ids_csv[i][0];
            int leaf_value = L_quant_csv[i][0];

            unsigned base_branch_id = tree_id * num_branches_per_tree;
            unsigned base_leaf_and_class_id = tree_id * num_branches_per_tree;
            unsigned branch_id_offset = i - first_ids_by_branch[tree_id];
            unsigned padded_branch_id = base_branch_id + branch_id_offset;

            for (unsigned v_id = 0; v_id < num_vars; v_id++) {
                BranchPair p;

                int low = T_quant_csv[i][v_id * 2];
                int high = T_quant_csv[i][v_id * 2 + 1];

                p.low = low;
                p.high = high;

                
                branches[padded_branch_id * num_vars + v_id] = p;
            }

            unsigned leaf_and_class_offset = branch_id_offset;
            unsigned padded_leaf_and_class_id = base_leaf_and_class_id + leaf_and_class_offset;

            //cout << "[Input::constructor]: padded_leaf_and_class_id = " << padded_leaf_and_class_id << endl;

            leaves[padded_leaf_and_class_id] = leaf_value;
            class_ids[padded_leaf_and_class_id] = class_id;
            tree_ids[padded_leaf_and_class_id] = tree_id;
        }

        for (unsigned i = 0; i < num_queries; i++) {
            for (unsigned v_id = 0; v_id < num_vars; v_id++) {
                queries[i * num_vars + v_id] = X_test_quant_csv[i][v_id];
            }
        }

        /*
        for (unsigned i = 0; i < num_branches_with_padding; i++) {
            for (unsigned v_id = 0; v_id < num_vars; v_id++) {
                cout << branches[i * num_vars + v_id] << " ";
            }

            cout << tree_ids[i] << " " << class_ids[i] << " " << leaves[i];
            cout << endl;
        }
        */
    }

    unsigned get_num_branches() {
        return num_branches;
    }

    unsigned get_num_dont_cares() {
        return dont_cares.size();
    }

    unsigned get_num_queries() {
        return num_queries;
    }

    unsigned get_num_iterations_required() {
        return num_trees / num_cores;
    }

    void add_dont_cares(unsigned var_id) {
        dont_cares.push_back(var_id);
    }

    vector<bitset<num_branches>> get_query_results(unsigned core_id = 0, int iteration_id = 0) {
        vector<bitset<num_branches>> result;
        result.resize(num_queries);
        unsigned b_id, v_id;

        //const unsigned num_branches_per_core = loaded_from_file ? (num_branches_with_padding / num_cores) : num_branches;
        const unsigned num_branches_per_core = num_branches;

        // TODO: Check if num_branches_per_core can be replaces with num_branches
        const unsigned branch_iteration_offset = iteration_id * num_cores * num_branches_per_core * num_vars;

/*
        cout << "[get_query_results]: Branches to be evaluated at (core_id, iteration_id) = (" << core_id << ", " << iteration_id << ")" << endl;
        for (b_id = 0; b_id < num_branches_per_core; b_id++) {
            unsigned b_offset = branch_iteration_offset + (core_id * num_branches_per_core + b_id) * num_vars;

            for (v_id = 0; v_id < num_vars; v_id++) {
                auto pair = branches[b_offset + v_id];

                cout << pair << " ";
            }
            cout << endl;
        }
*/

        //#pragma omp parallel for schedule(dynamic, 100)
        for (unsigned q_id = 0 ; q_id < num_queries; q_id++) {
            bitset<num_branches> query_set;
            unsigned q_offset = q_id * num_vars;

            for (b_id = 0; b_id < num_branches_per_core; b_id++) {
                unsigned b_offset = branch_iteration_offset + (core_id * num_branches_per_core + b_id) * num_vars;
                bool match = true;

                for (v_id = 0; v_id < num_vars; v_id++) {
                    //cout << "(b_id, v_id) = (" << b_id << ", " << v_id << "), " << endl;

                    unsigned val = queries[q_offset + v_id];
                    auto pair = branches[b_offset + v_id];

                    //cout << "(iteration_id, core_id, b_offset, v_id, pair) = (" << dec << iteration_id << ", " << core_id << ", " << b_offset << ", " << v_id << ", " << pair << "), " << endl;

                    if (dont_cares_set[v_id]) {
                        continue;
                    }

                    if (val < pair.low || val >= pair.high) {
                        match = false;
                        break;
                    }

                }

                query_set[b_id] = match;
                //cout << (match ? "matched" : "not matched") << endl;
            }

            result[q_id] = query_set;
        }

        return result;
    }

    vector<vector<bitset<num_branches>>> multi_core_get_query_results(int iteration_id = 0) {
        vector<vector<bitset<num_branches>>> result;

        for (unsigned i = 0; i < num_cores; i++) {
            result.push_back(get_query_results(i, iteration_id));
        }

        return result;
    }

    void print_contents() {
        cout << "(num_dont_cares, num_branches, num_queries) = (" << get_num_dont_cares() << ", " << get_num_branches() << ", " << get_num_queries() << ")" << endl; 

        cout << "[dont_cares]" << endl;
        for (auto dc: dont_cares) {
            cout << dc << ", ";
        }
        cout << endl;

        cout << "[leaf_values]" << endl;
        for (unsigned b_id = 0; b_id < num_branches * num_cores; b_id++) {
            cout << dec << leaves[b_id] << endl;
        }
        cout << endl;

        cout << "[branches]" << endl;
        for (unsigned b_id = 0; b_id < num_branches * num_cores; b_id++) {
            for (unsigned p_id = 0; p_id < num_vars; p_id++) {
                BranchPair pair = branches[b_id * num_vars + p_id];
                cout << "(" << pair.low << ", " << pair.high << "), ";
            }
            cout << endl;
        }

        cout << "[queries]" << endl;
        for (unsigned q_id = 0; q_id < num_queries; q_id++) {
            for (unsigned v_id = 0; v_id < num_vars; v_id++) {
                cout << queries[q_id * num_vars + v_id] << ", ";
            }
            cout << endl;
        }
    }

    unsigned fill_with_input_transactions(vector<int, aligned_allocator<int>> &source_input, bool enable_leaf_values = true, bool only_update_thresholds = false, int iteration_id = 0) {
        const unsigned num_macro_words_per_transaction = TX_SIZE / MACRO_WORD_SIZE;
        const unsigned num_dont_cares = get_num_dont_cares();

        //const unsigned num_branches_per_core = loaded_from_file ? (num_branches_with_padding / num_cores) : num_branches;
        const unsigned num_branches_per_core = num_branches;

        if (loaded_from_file) {
            // This ensures that branches are evenly split among cores
            assert(num_branches_with_padding % num_cores == 0);

            // This ensures that branches from any particular tree will
            // not be spread among more than one core, assuming that
            // no tree has more branches that `min_branches_per_match`.
            assert(num_branches_per_core % min_branches_per_match == 0);
        }

        const unsigned tx_boundary_conf = 0;
        const unsigned tx_boundary_thresholds = tx_boundary_conf + num_macro_words_per_transaction;
        const unsigned tx_boundary_dont_cares = tx_boundary_thresholds + num_macro_words_per_transaction * (2 * num_branches_per_core) * num_cores;
        const unsigned tx_boundary_queries = tx_boundary_dont_cares + (num_dont_cares > 0 ? num_macro_words_per_transaction : 0) * num_cores;
        const unsigned tx_boundary_final = tx_boundary_queries + num_macro_words_per_transaction * num_queries;

        assert(DATA_SIZE >= tx_boundary_final);
        assert(MACRO_WORD_SIZE % WORD_SIZE == 0);
        assert(TX_SIZE % MACRO_WORD_SIZE == 0);

        const unsigned num_vars_per_macro_word = MACRO_WORD_SIZE / WORD_SIZE;

        const unsigned num_vars_in_incomplete_pack = num_vars % num_vars_per_macro_word;
        const unsigned num_complete_packs = num_vars / num_vars_per_macro_word;
        const unsigned num_packs = num_complete_packs + (num_vars_in_incomplete_pack > 0 ? 1 : 0);

        const unsigned branch_iteration_offset = iteration_id * num_cores * num_branches_per_core * num_vars;
        const unsigned leaves_iteration_offset = iteration_id * num_cores * num_branches_per_core;

        source_input[tx_boundary_conf + 0] = num_branches_per_core;
        source_input[tx_boundary_conf + 1] = num_dont_cares;
        source_input[tx_boundary_conf + 2] = num_queries;
        source_input[tx_boundary_conf + 3] = num_vars;
        source_input[tx_boundary_conf + 4] = enable_leaf_values;
        source_input[tx_boundary_conf + 5] = num_cores;

        for (unsigned c_id = 0; c_id < num_cores; c_id++) {
            for (unsigned b_id = 0; b_id < num_branches_per_core; b_id++) {
                unsigned v_id = 0;

                for (unsigned pack_id = 0; pack_id < num_packs; pack_id++) {
                    unsigned packed_variables_low = 0;
                    unsigned packed_variables_high = 0;
                    
                    for (unsigned sub_pack_id = 0; sub_pack_id < num_vars_per_macro_word; sub_pack_id++) {
                        BranchPair pair = branches[branch_iteration_offset + (c_id * num_branches_per_core + b_id) * num_vars + v_id];

                        if (pack_id == num_complete_packs && sub_pack_id == num_vars_in_incomplete_pack) {
                            break;
                        }

                        packed_variables_low = packed_variables_low | (pair.low << (sub_pack_id * WORD_SIZE));
                        packed_variables_high = packed_variables_high | (pair.high << (sub_pack_id * WORD_SIZE));
                        
                        v_id++;                        
                    }

                    source_input[tx_boundary_thresholds + ((num_branches_per_core * c_id + b_id) * 2) * num_macro_words_per_transaction + pack_id] = packed_variables_low;
                    source_input[tx_boundary_thresholds + ((num_branches_per_core * c_id + b_id) * 2 + 1) * num_macro_words_per_transaction + pack_id] = packed_variables_high;
                }

                // The tree and class ids are encoded in low-threshold configuration lines
                source_input[tx_boundary_thresholds + ((num_branches_per_core * c_id + b_id) * 2) * num_macro_words_per_transaction + (num_macro_words_per_transaction - 1)] = (class_ids[leaves_iteration_offset + num_branches_per_core * c_id + b_id] << TREE_ID_NUM_BITS) | tree_ids[leaves_iteration_offset + num_branches_per_core * c_id + b_id];

                // The leaf values are encoded in high-threshold configuration lines
                source_input[tx_boundary_thresholds + ((num_branches_per_core * c_id + b_id) * 2 + 1) * num_macro_words_per_transaction + (num_macro_words_per_transaction - 1)] = leaves[leaves_iteration_offset + num_branches_per_core * c_id + b_id];
            }
        }

        if (!only_update_thresholds) {
            // TODO: Let don't care variables be encoded in a more compact way, so that more
            //       don't care variables might be specified. Note that, right now, the
            //       maximum number of don't cares supported by the system is
            //       (transaction_size / macro_word_size)
            for (unsigned c_id = 0; c_id < num_cores; c_id++) {
                for (unsigned dc_id = 0; dc_id < num_dont_cares; dc_id++) {
                    source_input[tx_boundary_dont_cares + c_id * num_macro_words_per_transaction + dc_id] = dont_cares[dc_id];
                }
            }

            for (unsigned q_id = 0; q_id < num_queries; q_id++) {
                unsigned v_id = 0;

                for (unsigned pack_id = 0; pack_id < num_packs; pack_id++) {
                    unsigned packed_variables = 0;
                    
                    for (unsigned sub_pack_id = 0; sub_pack_id < num_vars_per_macro_word; sub_pack_id++) {
                        if (pack_id == num_complete_packs && sub_pack_id == num_vars_in_incomplete_pack) {
                            break;
                        }

                        // Here we are assuming that elements of `queries` do
                        // not have asserted bits at positions >= WORD_SIZE .
                        // Also, notice that input variables and thresholds
                        // are unsigned.
                        packed_variables = packed_variables | (queries[q_id * num_vars + v_id] << (sub_pack_id * WORD_SIZE));
                        
                        v_id++;                        
                    }

                    source_input[tx_boundary_queries + q_id * num_macro_words_per_transaction + pack_id] = packed_variables;
                }
            }
            
            // TODO: REMOVE
            //cout << "[input_stream]" << endl;
            //nice_print_vector(source_input, tx_boundary_final);

            return tx_boundary_final;
        } else {
            return tx_boundary_dont_cares;
        }
    }

    unsigned fill_with_output_transactions(vector<int, aligned_allocator<int>> &ref_output, bool enable_leaf_values = true, bool noc_test = true, int iteration_id = 0) {
        //vector<bitset<num_branches>> results = get_query_results();

        // Outer vector: num_queries entries, where each entry j
        //               corresponds to a particular query input
        // Inner vector: num_cores entries, where entry i contains
        //               is the set of branch matches of core i
        //               for query j
        cout << "[fill_with_output_transactions] iteration_id: " << iteration_id << endl;
        vector<vector<bitset<num_branches>>> results = multi_core_get_query_results(iteration_id);

        //const unsigned num_branches_per_core = loaded_from_file ? (num_branches_with_padding / num_cores) : num_branches;
        const unsigned num_branches_per_core = num_branches;

        if (loaded_from_file) {
            // This ensures that branches are evenly split among cores
            assert(num_branches_with_padding % num_cores == 0);

            // This ensures that branches from any particular tree will
            // not be spread among more than one core, assuming that
            // no tree has more branches that `min_branches_per_match`.
            assert(num_branches_per_core % min_branches_per_match == 0);
        }

        const unsigned num_macro_words_per_transaction = TX_SIZE / MACRO_WORD_SIZE;

        const unsigned branch_iteration_offset = iteration_id * num_cores * num_branches_per_core * num_vars;
        const unsigned leaves_iteration_offset = iteration_id * num_cores * num_branches_per_core;

        assert(sizeof(unsigned long) * 8 == 64);
        unsigned long out_i = 0;
        unsigned long t_i = 0;

        if (enable_leaf_values) {
            if (noc_test) {
                vector<int> aux_leaf_values;

                // The following will round the division up
                unsigned num_leaf_value_slots = (num_branches_per_core + min_branches_per_match - 1) / min_branches_per_match;

                aux_leaf_values.resize(num_leaf_value_slots);

                for (unsigned q_i = 0; q_i < num_queries; q_i++) {
                    int total_leaf_value = 0;

                    for (unsigned c_i = 0; c_i < num_cores; c_i++) {
                        for (unsigned i = 0; i < num_leaf_value_slots; i++) {
                            aux_leaf_values[i] = 0;                                                
                        }

                        for (unsigned i = 0; i < num_branches_per_core; i++) {
                            //cout << "(t_i, out_i, i / min_branches_per_match) = (" << t_i << ", " << out_i << ", " << i / min_branches_per_match << ")" << endl;
                            if (results[c_i][q_i][i]) {
                                aux_leaf_values[i / min_branches_per_match] = leaves[leaves_iteration_offset + c_i * num_branches_per_core + i];
                                //cout << "Matched leaf value: " << leaves[leaves_iteration_offset + c_i * num_branches_per_core + i] << endl;
                            }
                            t_i += 1;
                        }

                        for (unsigned i = 0; i < num_leaf_value_slots; i++) {
                            total_leaf_value += aux_leaf_values[i];                                                
                        }
                    }

                    unsigned class_id = total_leaf_value > 0 ? 1 : 0;

                    ref_output[out_i] = total_leaf_value;
                    ref_output[out_i + 1] = (class_id << TREE_ID_NUM_BITS) | 0;

                    out_i += num_macro_words_per_transaction;
                }
            } else {
                for (auto set: results[0]) {
                    for (unsigned i = 0; i < num_branches_per_core; i++) {
                        //cout << "(t_i, out_i, i / min_branches_per_match) = (" << t_i << ", " << out_i << ", " << i / min_branches_per_match << ")" << endl;
                        if (set[i]) {
                            ref_output[out_i + (i / min_branches_per_match) * 2] = leaves[leaves_iteration_offset + i];
                            ref_output[out_i + (i / min_branches_per_match) * 2 + 1] = (class_ids[leaves_iteration_offset + i] << TREE_ID_NUM_BITS) | tree_ids[leaves_iteration_offset + i];
                        }
                        t_i += 1;
                    }
                    out_i += num_macro_words_per_transaction;
                }
            }
        } else {
            const bitset<num_branches> mask_32(0xffffffff);
            
            for (auto set: results[0]) {
                // TODO-PERFORMANCE: Possibly make this faster by writing two 32-bit
                //                   words at a time.
                //                   Also, avoid checking positions beyond `num_branches`
                for (int i = 0; i < TX_SIZE / 32; i++) {
                    unsigned word = static_cast<unsigned>((set & mask_32).to_ulong());
                    ref_output[out_i] = word;
                    set >>= 32;
                    out_i++;
                }
            }
        }

#ifdef VERBOSE
        cout << "[output_stream]" << endl;
        nice_print_vector(ref_output, out_i);
#endif

        return (TX_SIZE / MACRO_WORD_SIZE) * results[0].size();
    }
};

int program_and_execute(string binaryFile) {
    // TODO: Dynamically calculate this value
    auto size = DATA_SIZE;
    // Allocate Memory in Host Memory
    auto vector_size_bytes = sizeof(int) * size;
    vector<int, aligned_allocator<int> > source_input1(size);
    vector<int, aligned_allocator<int> > source_input2(size);
    vector<int, aligned_allocator<int> > source_hw_results(size);
    vector<int, aligned_allocator<int> > source_sw_results(size);
    vector<int, aligned_allocator<int> > accumulated_results(size);

    /*
    // Create the test data and Software Result
    for (int i = 0; i < size; i++) {
        //source_input1[i] = i;
        //source_input2[i] = i;
        //source_sw_results[i] = source_input1[i] + source_input2[i];
        //source_hw_results[i] = 0;
    }
    */

    unsigned num_input_words = 0;
    unsigned num_output_words = 0;

    //const unsigned num_branches = 256;
    //const unsigned num_branches = 42;
    const unsigned num_branches = 32;

    // TODO: LET THIS SUPPORT VALUES
    //       GREATER THAN 16
    //const unsigned num_variables = 120;
    //const unsigned num_variables = 115;
    //const unsigned num_variables = 43;
    //const unsigned num_variables = 31;
    //const unsigned num_variables = 30;
    const unsigned num_variables = 29;
    //const unsigned num_variables = 18;
    //const unsigned num_variables = 17;
    //const unsigned num_variables = 16;
    //const unsigned num_variables = 15;
    //const unsigned num_variables = 4;
    //const unsigned num_variables = 3;
    const unsigned num_dont_cares = 3;
    //const unsigned num_dont_cares = 0;
    //const unsigned max_state = 1 << 16 - 1;
    //const unsigned max_state = 4095;
    //const unsigned max_state = 255;
    const unsigned max_state = 15;
    //const unsigned max_state = 14;
    //const unsigned num_queries = 100000000; // This might trigger a bad alloc
    //const unsigned num_queries = 10000000;
    //const unsigned num_queries = 1000000;
    //const unsigned num_queries = 100000;
    //const unsigned num_queries = 20000;
    //const unsigned num_queries = 10000;
    const unsigned num_queries = 5000;
    //const unsigned num_queries = 1024;
    //const unsigned num_executions = 3;
    const unsigned num_executions = 3;
    //const unsigned min_branches_per_match = 256;
    //const unsigned min_branches_per_match = 128;
    const unsigned min_branches_per_match = 16;
    //const unsigned min_branches_per_match = 8;
    //const unsigned num_cores = 8;
    //const unsigned num_cores = 4;
    const unsigned num_cores = 2;
    //const unsigned num_cores = 2;
    //const unsigned num_cores = 1;
    //const unsigned num_queries = 512;
    //const unsigned num_queries = 256;
    //const unsigned num_queries = 128;
    //const unsigned num_queries = 64;
    //const unsigned num_queries = 32;
    //const unsigned num_queries = 16;
    //const unsigned num_queries = 4;
    //const unsigned num_queries = 1;
    const bool enable_leaf_values = true;
    const bool enable_compact_trees = false;

    //init_compact_id(num_branches);
    //return 0;

    //Input<num_branches, num_variables, max_state> random_input(num_dont_cares, num_queries, 0.84f);
    //Input<num_branches, num_variables, max_state> random_input(num_dont_cares, num_queries, 0.92f, enable_leaf_values, min_branches_per_match, num_cores, enable_compact_trees);
    //Input<num_branches, num_variables, max_state> random_input(num_dont_cares, num_queries, 0.75f, enable_leaf_values, min_branches_per_match, num_cores, enable_compact_trees);
    Input<num_branches, num_variables, max_state> random_input(num_dont_cares, num_queries, 0.87f, enable_leaf_values, min_branches_per_match, num_cores, enable_compact_trees);
    random_input.print_contents();
    cout << "Just completed input generation" << endl;

    num_input_words = random_input.fill_with_input_transactions(source_input1, enable_leaf_values);
    cout << "Just completed loading to source_input1" << endl;
    num_output_words = random_input.fill_with_output_transactions(source_sw_results, enable_leaf_values);
    cout << "Just completed loading to source_sw_results" << endl;

#ifdef VERBOSE
    cout << "[main]: source_input1 data:" << endl;
    for (int i = 0; i < num_input_words; i++) {
        if (i % 16 == 0 && i > 0) {
            cout << endl;
        }
        cout << "0x" << hex << source_input1[i] << ", ";
    }
    cout << endl;
#endif

    // OPENCL HOST CODE AREA START
    // Create Program and Kernel
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_vadd;
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            cout << "Failed to program device[" << i << "] with xclbin file!" << endl;
        } else {
            cout << "Device[" << i << "]: program successful!" << endl;
            OCL_CHECK(err, krnl_vadd = cl::Kernel(program, "krnl_vadd_rtl", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        cout << "Failed to program any device found, exit!" << endl;
        exit(EXIT_FAILURE);
    }

    unsigned original_num_input_words = num_input_words;

    long long int total_time = 0;

    int match0;
    int match1;

    for (unsigned i = 0; i < num_executions; i++) {
        // Allocate Buffer in Global Memory
        /*
        OCL_CHECK(err, cl::Buffer buffer_r1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                            source_input1.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_r2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                            source_input2.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes,
                                           source_hw_results.data(), &err));
       */

        if (i > 0) {
            random_input.update_thresholds();
            cout << "Just finished generating new batch of thresholds" << endl;
            num_input_words = random_input.fill_with_input_transactions(source_input1, enable_leaf_values, true);
            cout << "Just completed loading updated thresholds to source_input1" << endl;
            random_input.fill_with_output_transactions(source_sw_results, enable_leaf_values);
        }

        OCL_CHECK(err, cl::Buffer buffer_r1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, original_num_input_words * 4,
                                            source_input1.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_r2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, original_num_input_words * 4,
                                            source_input2.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num_output_words * 4,
                                           source_hw_results.data(), &err));

        // Set the Kernel Arguments
        OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_r1));
        OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_r2));
        OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_w));
        //OCL_CHECK(err, err = krnl_vadd.setArg(3, size / 16));

        // We divide by 16 here because this controls the number of 64-byte
        // transactions performed by the system
        OCL_CHECK(err, err = krnl_vadd.setArg(3, original_num_input_words / 16));

        cout << "Waiting for user input to continue..." << endl;

        cout << "Sending " << std::dec << num_input_words / 16 << " transactions" << endl;
        cout << "Expecting " << num_output_words / 16 << " transactions to be sent back" << endl;

        //cin.ignore(800, '\n');

        cout << "Continuing!" << endl;

        auto pre_begin = chrono::high_resolution_clock::now();

        // Copy input data to device global memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_r1, buffer_r2}, 0 /* 0 means from host*/));

        auto begin = chrono::high_resolution_clock::now();

        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_vadd));

        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_w}, CL_MIGRATE_MEM_OBJECT_HOST));

        OCL_CHECK(err, err = q.finish());

        auto end = chrono::high_resolution_clock::now();

        auto acc_begin = chrono::high_resolution_clock::now();

        // This ensures that we only parallelize the code
        // when there are enough queries to justify the
        // thread spawning overhead.
        if (num_queries > 1000000) {
            #pragma omp parallel for num_threads(8)
            for (unsigned i = 0; i < num_queries; i++) {
                accumulated_results[i] += source_hw_results[i*16]; 
            }
        } else {
            for (unsigned i = 0; i < num_queries; i++) {
                accumulated_results[i] += source_hw_results[i*16]; 
            }
        }

        auto acc_end = chrono::high_resolution_clock::now();

        auto elapsed_time = chrono::duration_cast<chrono::nanoseconds>(end-begin).count();
        auto elapsed_time_with_loading = chrono::duration_cast<chrono::nanoseconds>(end-pre_begin).count();
        auto accumulation_time = chrono::duration_cast<chrono::nanoseconds>(acc_end-acc_begin).count();

        total_time += elapsed_time_with_loading;
        total_time += accumulation_time;

        auto freq = ((double) 1000000000 * num_queries) / elapsed_time;
        auto freq_with_loading = ((double) 1000000000 * num_queries) / elapsed_time_with_loading;
        auto acc_freq = ((double) 1000000000 * num_queries) / accumulation_time;

        cout << "[main] Kernel execution time: " << elapsed_time << " ns" << endl;
        cout << "[main] Query processing rate: " << freq << " queries/sec" << endl;
        cout << "[main] Query accumulation rate: " << acc_freq << " queries/sec" << endl;
        cout << "[main] Query processing rate (including host-FPGA data copy): " << freq_with_loading << " queries/sec" << endl;

        // OPENCL HOST CODE AREA END

    #ifdef VERBOSE
        // Compare the results of the Device to the simulation
        cout << "[main]: source_hw_results data:" << endl;
        for (int i = 0; i < num_output_words; i++) {
            if (i % 16 == 0 && i > 0) {
                cout << endl;
            }
            //cout << "0x" << hex << source_hw_results[i] << ", ";
            cout << "0x" << std::setfill ('0') << std::setw(sizeof(unsigned)*2) << hex << source_hw_results[i] << ", ";
        }
    #endif

        cout << endl;

        match0 = 0;
        match1 = 0;

        for (unsigned i = 0; i < num_output_words; i++) {
            if (source_hw_results[i] != source_sw_results[i]) {
                match0 = 1;
                break;
            }
        }

        unsigned num_macro_words_per_transaction = TX_SIZE / MACRO_WORD_SIZE;
        if (match0) {
            // We do a subtraction here because, in this scenario,
            // the last transaction cannot be retrieved.
            for (unsigned i = 0; i < (num_output_words - num_macro_words_per_transaction); i++) {
                if (source_hw_results[i+16] != source_sw_results[i]) {
                    match1 = 1;
                    break;
                }
            }
        } else {
            match1 = 1;
        }

        if (!match0) {
            cout << "Output correctly generated at transaction offset: 0" << endl;
        }

        if (!match1) {
            cout << "Output correctly generated at transaction offset: 1" << endl;
        }

        if (match0 && match1) {
            cout << "First offset-0 error:" << endl;
            for (unsigned i = 0; i < num_output_words; i++) {
                if (source_hw_results[i] != source_sw_results[i]) {
                    cout << "Error: Result mismatch" << endl;
                    cout << "i = " << i << " Software result = " << source_sw_results[i]
                         << " Device result = " << source_hw_results[i] << endl;
                    match0 = 1;
                    cout << "[relevant query]" << endl;

                    for (unsigned j = 0; j < num_variables; j++) {
                        unsigned offset = i / 16;
                        cout << random_input.queries[j + offset * num_variables] << ", ";
                    }

                    cout << endl;

                    break;
                }
            }

            cout << "First offset-1 error:" << endl;
            for (unsigned i = 0; i < num_output_words; i++) {
                if (source_hw_results[i+16] != source_sw_results[i]) {
                    cout << "Error: Result mismatch" << endl;
                    cout << "i = " << i << " Software result = " << source_sw_results[i]
                         << " Device result = " << source_hw_results[i+16] << endl;
                    cout << "[relevant query]" << endl;

                    for (unsigned j = 0; j < num_variables; j++) {
                        unsigned offset = i / 16;
                        cout << random_input.queries[j + offset * num_variables] << ", ";
                    }

                    cout << endl;

                    break;
                }
            }
        }
    }

    auto average_freq = ((double) 1000000000 * num_queries) / total_time;

    cout << "[main] Total query processing time: " << total_time << " ns" << endl;
    cout << "[main] Average query processing rate: " << average_freq << " queries/sec" << endl;

    cout << "[main]: Accumulated data" << endl;
    for (unsigned i = 0; i < num_queries; i++) {
        cout << "accumulated_results[ " << i << "]: " << accumulated_results[i] << endl;
    }

    return (match0 && match1 ? EXIT_FAILURE : EXIT_SUCCESS);
}

int test_csv_parsing(string basePath) {
    read_csv(basePath, "T_quant.csv");
    return 0;
}

double evaluate_accuracy(string basePath, vector<int, aligned_allocator<int>> accumulated_results) {
    vector<vector<int>> y_test_csv = read_csv(basePath, "y_test.csv");
    int num_queries = y_test_csv.size();
    int matches = 0;

    for (int i = 0; i < num_queries; i++) {
        int predicted = accumulated_results[i] < 0 ? 0 : 1;
        int reference = y_test_csv[i][0];

        if (predicted == reference) {
            matches += 1;
        }
    }

    cout << "matches: " << matches << endl;

    return 1.0f * matches / num_queries;
}

int test_pushing_csv_to_fpga(string binaryFile, string basePath) {
    auto size = DATA_SIZE;
    auto vector_size_bytes = sizeof(int) * size;
    vector<int, aligned_allocator<int> > source_input1(size);
    vector<int, aligned_allocator<int> > source_input2(size);
    vector<int, aligned_allocator<int> > source_hw_results(size);
    //vector<int, aligned_allocator<int> > *source_hw_results = new vector<int, aligned_allocator<int>>(size);
    vector<int, aligned_allocator<int> > source_sw_results(size);
    vector<int, aligned_allocator<int> > accumulated_results(size);
    //vector<int, aligned_allocator<int> > *accumulated_results = new vector<int, aligned_allocator<int>>(size);

    unsigned num_input_words = 0;
    unsigned num_output_words = 0;

    //const unsigned num_branches = 256;
    //const unsigned num_branches = 42;
    //const unsigned num_branches = 32;
    const unsigned num_branches = 4;
    //const unsigned max_num_vars = 120;
    //const unsigned max_num_vars = 115;
    //const unsigned max_num_vars = 43;
    //const unsigned max_num_vars = 31;
    //const unsigned max_num_vars = 30;
    const unsigned max_num_vars = 29;
    //const unsigned max_num_vars = 18;
    //const unsigned max_num_vars = 17;
    //const unsigned max_num_vars = 16;
    //const unsigned max_num_vars = 15;
    //const unsigned max_num_vars = 19;
    //const unsigned max_num_vars = 12;
    //const unsigned max_num_vars = 4;
    //const unsigned max_num_vars = 3;
    //const unsigned num_dont_cares = 3;
    //const unsigned num_dont_cares = 0;
    //const unsigned max_state = 1 << 16 - 1;
    //const unsigned max_state = 4095;
    const unsigned max_state = 255;
    //const unsigned max_state = 15;
    //const unsigned max_state = 14;
    //const unsigned min_branches_per_match = 256;
    //const unsigned min_branches_per_match = 128;
    //const unsigned min_branches_per_match = 256;
    //const unsigned min_branches_per_match = 8;
    const unsigned min_branches_per_match = 4;
    //const unsigned num_cores = 256;
    const unsigned num_cores = 128;
    //const unsigned num_cores = 8;
    //const unsigned num_cores = 4;
    //const unsigned num_cores = 1;
    //const unsigned num_queries = 512;
    //const unsigned num_queries = 256;
    //const unsigned num_queries = 128;
    //const unsigned num_queries = 64;
    //const unsigned num_queries = 32;
    //const unsigned num_queries = 16;
    //const unsigned num_queries = 4;
    //const unsigned num_queries = 1;
    const bool enable_leaf_values = true;
    const bool enable_compact_trees = false;

    Input<num_branches, max_num_vars, max_state> input_from_csv(basePath, enable_leaf_values, min_branches_per_match, num_cores, enable_compact_trees);
    const unsigned num_queries = input_from_csv.num_queries;
    const unsigned num_variables = input_from_csv.num_vars;
    const unsigned num_iterations_required = input_from_csv.get_num_iterations_required();

    //input_from_csv.print_contents();
    cout << "Num iterations required: " << num_iterations_required << endl;

    cout << "Just completed input generation" << endl;

    num_input_words = input_from_csv.fill_with_input_transactions(source_input1, enable_leaf_values);
    cout << "Just completed loading to source_input1" << endl;
    num_output_words = input_from_csv.fill_with_output_transactions(source_sw_results, enable_leaf_values);
    cout << "Just completed loading to source_sw_results" << endl;

#ifdef VERBOSE
    cout << "[main]: source_input1 data:" << endl;
    for (int i = 0; i < num_input_words; i++) {
        if (i % 16 == 0 && i > 0) {
            cout << endl;
        }
        cout << "0x" << hex << source_input1[i] << ", ";
    }
    cout << endl;
#endif

    // OPENCL HOST CODE AREA START
    // Create Program and Kernel
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_vadd;
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            cout << "Failed to program device[" << i << "] with xclbin file!" << endl;
        } else {
            cout << "Device[" << i << "]: program successful!" << endl;
            OCL_CHECK(err, krnl_vadd = cl::Kernel(program, "krnl_vadd_rtl", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        cout << "Failed to program any device found, exit!" << endl;
        exit(EXIT_FAILURE);
    }

    unsigned original_num_input_words = num_input_words;

    long long int total_time = 0;

    int match0;
    int match1;

    for (unsigned i = 0; i < num_iterations_required; i++) {
        if (i > 0) {
            //input_from_csv.update_thresholds();
            cout << "Just finished generating new batch of thresholds" << endl;
            num_input_words = input_from_csv.fill_with_input_transactions(source_input1, enable_leaf_values, true, i);
            cout << "Just completed loading updated thresholds to source_input1" << endl;
            input_from_csv.fill_with_output_transactions(source_sw_results, enable_leaf_values, true, i);
        }

        OCL_CHECK(err, cl::Buffer buffer_r1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num_input_words * 4,
                                            source_input1.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_r2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num_input_words * 4,
                                            source_input2.data(), &err));
        OCL_CHECK(err, cl::Buffer buffer_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num_output_words * 4,
                                           source_hw_results.data(), &err));

        // Set the Kernel Arguments
        OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_r1));
        OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_r2));
        OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_w));
        //OCL_CHECK(err, err = krnl_vadd.setArg(3, size / 16));

        // We divide by 16 here because this controls the number of 64-byte
        // transactions performed by the system
        OCL_CHECK(err, err = krnl_vadd.setArg(3, original_num_input_words / 16));

        cout << "Waiting for user input to continue..." << endl;

        cout << "Sending " << std::dec << num_input_words / 16 << " transactions" << endl;
        cout << "Expecting " << num_output_words / 16 << " transactions to be sent back" << endl;

        //cin.ignore(800, '\n');

        cout << "Continuing!" << endl;

        auto pre_begin = chrono::high_resolution_clock::now();

        // Copy input data to device global memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_r1, buffer_r2}, 0)); // 0 means from host

        auto begin = chrono::high_resolution_clock::now();

        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_vadd));

        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_w}, CL_MIGRATE_MEM_OBJECT_HOST));

        OCL_CHECK(err, err = q.finish());

        auto end = chrono::high_resolution_clock::now();

        auto acc_begin = chrono::high_resolution_clock::now();

        // This ensures that we only parallelize the code
        // when there are enough queries to justify the
        // thread spawning overhead.
        if (num_queries > 1000000) {
            #pragma omp parallel for num_threads(8)
            for (unsigned i = 0; i < num_queries; i++) {
                accumulated_results[i] += source_hw_results[i*16]; 
            }
        } else {
            for (unsigned i = 0; i < num_queries; i++) {
                accumulated_results[i] += source_hw_results[i*16]; 
            }
        }

        for (unsigned i = 0; i < num_queries; i++) {
            accumulated_results[i] += source_hw_results[i*16]; 
        }

        auto acc_end = chrono::high_resolution_clock::now();

        auto elapsed_time = chrono::duration_cast<chrono::nanoseconds>(end-begin).count();
        auto elapsed_time_with_loading = chrono::duration_cast<chrono::nanoseconds>(end-pre_begin).count();
        auto accumulation_time = chrono::duration_cast<chrono::nanoseconds>(acc_end-acc_begin).count();

        total_time += elapsed_time_with_loading;
        total_time += accumulation_time;

        auto freq = ((double) 1000000000 * num_queries) / elapsed_time;
        auto freq_with_loading = ((double) 1000000000 * num_queries) / elapsed_time_with_loading;
        auto acc_freq = ((double) 1000000000 * num_queries) / accumulation_time;

        cout << "[main] Kernel execution time: " << elapsed_time << " ns" << endl;
        cout << "[main] Query processing rate: " << freq << " queries/sec" << endl;
        cout << "[main] Query accumulation rate: " << acc_freq << " queries/sec" << endl;
        cout << "[main] Query processing rate (including host-FPGA data copy): " << freq_with_loading << " queries/sec" << endl;

        // OPENCL HOST CODE AREA END

    #ifdef VERBOSE
        // Compare the results of the Device to the simulation
        cout << "[main]: source_hw_results data:" << endl;
        for (int i = 0; i < num_output_words; i++) {
            if (i % 16 == 0 && i > 0) {
                cout << endl;
            }
            //cout << "0x" << hex << source_hw_results[i] << ", ";
            cout << "0x" << std::setfill ('0') << std::setw(sizeof(unsigned)*2) << hex << source_hw_results[i] << ", ";
        }
    #endif

        cout << endl;

        match0 = 0;
        match1 = 0;

        for (unsigned i = 0; i < num_output_words; i++) {
            if (source_hw_results[i] != source_sw_results[i]) {
                match0 = 1;
                break;
            }
        }

        unsigned num_macro_words_per_transaction = TX_SIZE / MACRO_WORD_SIZE;
        if (match0) {
            // We do a subtraction here because, in this scenario,
            // the last transaction cannot be retrieved.
            for (unsigned i = 0; i < (num_output_words - num_macro_words_per_transaction); i++) {
                if (source_hw_results[i+16] != source_sw_results[i]) {
                    match1 = 1;
                    break;
                }
            }
        } else {
            match1 = 1;
        }

        if (!match0) {
            cout << "Output correctly generated at transaction offset: 0" << endl;
        }

        if (!match1) {
            cout << "Output correctly generated at transaction offset: 1" << endl;
        }

        if (match0 && match1) {
            cout << "First offset-0 error:" << endl;
            for (unsigned i = 0; i < num_output_words; i++) {
                if (source_hw_results[i] != source_sw_results[i]) {
                    cout << "Error: Result mismatch" << endl;
                    cout << "i = " << i << " Software result = " << source_sw_results[i]
                         << " Device result = " << source_hw_results[i] << endl;
                    match0 = 1;
                    cout << "[relevant query]" << endl;

                    for (unsigned j = 0; j < num_variables; j++) {
                        unsigned offset = i / 16;
                        cout << input_from_csv.queries[j + offset * num_variables] << ", ";
                    }

                    cout << endl;

                    break;
                }
            }

            cout << "First offset-1 error:" << endl;
            for (unsigned i = 0; i < num_output_words; i++) {
                if (source_hw_results[i+16] != source_sw_results[i]) {
                    cout << "Error: Result mismatch" << endl;
                    cout << "i = " << i << " Software result = " << source_sw_results[i]
                         << " Device result = " << source_hw_results[i+16] << endl;
                    cout << "[relevant query]" << endl;

                    for (unsigned j = 0; j < num_variables; j++) {
                        unsigned offset = i / 16;
                        cout << input_from_csv.queries[j + offset * num_variables] << ", ";
                    }

                    cout << endl;

                    break;
                }
            }
        }
    }

    auto average_freq = ((double) 1000000000 * num_queries) / total_time;

    cout << "[main] Total query processing time: " << total_time << " ns" << endl;
    cout << "[main] Average query processing rate: " << average_freq << " queries/sec" << endl;

/*
    cout << "[main]: Accumulated data" << endl;
    for (unsigned i = 0; i < num_queries; i++) {
        cout << "accumulated_results[ " << i << "]: " << accumulated_results[i] << endl;
    }
*/

    cout << "[main]: Accuracy: " << evaluate_accuracy(basePath, accumulated_results) << endl;

    return (match0 && match1 ? EXIT_FAILURE : EXIT_SUCCESS);
}

int test_input_creation_from_csv(string basePath) {
    // TODO: Avoid the need to set these template parameters here
    Input<256, 3, 15> input_from_csv(basePath);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <XCLBIN File>" << " <QUANTIZED DATASET BASE PATH>" << endl;
        return EXIT_FAILURE;
    }

    string binaryFile = argv[1];
    string basePath = argv[2];

    //return program_and_execute(binaryFile);
    //return test_csv_parsing(basePath);
    //return test_input_creation_from_csv(basePath);
    return test_pushing_csv_to_fpga(binaryFile, basePath);
}
