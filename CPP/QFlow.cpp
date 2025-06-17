#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib> // For rand

void generate_combinations(const std::vector<int>& input, int r, std::vector<std::vector<int>>& output) {
    std::vector<bool> bitmask(r, true);  // r ones
    bitmask.resize(input.size(), false); // followed by n-r zeros

    do {
        std::vector<int> combination;
        for (size_t i = 0; i < input.size(); ++i) {
            if (bitmask[i]) {
                combination.push_back(input[i]);
            }
        }
        output.push_back(combination);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

int main() {
    using IndexVector = std::vector<int>;

    int nao    = 3; // Number of active occupied orbitals  == n_occ_alpha
    int nav    = 3; // Number of active virtual orbitals == n_virt_alpha
    int nocc   = 5; // Total number of occupied orbitals
    int nvirt  = 5; // Total number of virtual orbitals
    int cycles = 1; // Number of cycles to run

    // Skip this: This creates a random eval_sorted
    std::vector<double> eval_sorted(nocc+nvirt);
    for (int i = 0; i < (nocc+nvirt); i++)
        eval_sorted[i] = rand() % 12345;
    std::sort(eval_sorted.begin(), eval_sorted.end());
    //End Skip



    // STEP 1: Create master T1 and T2 Tensors (Global Pool)


    // STEP 2: Create a list of occupied and virtual combinations 
    //         of orbital and then combine those in all ways. 
    std::vector<int> occ_list, virt_list;
    for (int i = 0; i < nocc; ++i)
        occ_list.push_back(i);
    for (int i = nocc; i < nocc + nvirt; ++i)
        virt_list.push_back(i);

    std::vector<std::vector<int>> occ_combinations;
    std::vector<std::vector<int>> virt_combinations;
    generate_combinations(occ_list, nao, occ_combinations);
    generate_combinations(virt_list, nav, virt_combinations);

    // Store combinations with their orb_e_diff
    std::vector<std::pair<std::vector<int>, double>> all_combinations_with_energy;

    for (const auto& occ_set : occ_combinations) {
        for (const auto& virt_set : virt_combinations) {
            std::vector<int> combined;
    
            // First nao: original occ_set
            combined.insert(combined.end(), occ_set.begin(), occ_set.end());
    
            // Next nao: occ_set + nocc
            for (int val : occ_set)
                combined.push_back(val + nocc);
    
            // Next nav: virt_set + nocc
            for (int val : virt_set)
                combined.push_back(val + nocc);
    
            // Final nav: virt_set + nocc + nvirt
            for (int val : virt_set)
                combined.push_back(val + nocc + nvirt);
    
            double occ_sum = 0.0, virt_sum = 0.0;
            for (int j : occ_set)
                occ_sum += eval_sorted[j];
            for (int i : virt_set)
                virt_sum += eval_sorted[i];
    
            double orb_e_diff = virt_sum - occ_sum;
            all_combinations_with_energy.emplace_back(combined, orb_e_diff);
        }
    }

    // Sort combinations based on orb_e_diff
    std::sort(all_combinations_with_energy.begin(), all_combinations_with_energy.end(),
    [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    // DEBUG: print a few sorted results
    for (size_t i = 0; i < std::min(size_t(10), all_combinations_with_energy.size()); ++i) {
        const auto& [combo, diff] = all_combinations_with_energy[i];
        std::cout << "orb_e_diff: " << diff << " | combo: ";
        for (int idx : combo) std::cout << idx << " ";
        std::cout << "\n";
    }

    // Extract sorted combinations only
    std::vector<std::vector<int>> sorted_combinations;
    for (const auto& [combo, _] : all_combinations_with_energy) {
        sorted_combinations.push_back(combo);
    }

    std::cout << "Total combinations: " << sorted_combinations.size() << std::endl;


    for (int cycle = 0; cycle < cycles; ++cycle) {
        // STEP 3: 
        std::cout << "Cycle " << cycle + 1 << " of " << cycles << std::endl;
        
        for (const auto& combination : sorted_combinations) {
            std::cout << "Combination: ";
            for (int idx : combination) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;

            IndexVector occ_int_vec(combination.begin(), combination.begin() + 2*nao);
            IndexVector virt_int_vec(combination.end() - 2*nav, combination.end());
            
            std::cout << "occ_int_vec: ";
            for (int idx : occ_int_vec) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;

            std::cout << "virt_int_vec: ";
            for (int idx : virt_int_vec) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;

            // Here you would call DUCC with the indices:
            // IndexSpace MO_IS{
            //     {"occ_int_vec", occ_int_vec},
            //     {"virt_int_vec", virt_int_vec},
            // }

            // DUCC will print the active space Hamiltonian. For simplicity, the printing
            // will be changed to all elements of hte AS Hamiltonian since we will be 
            // dealing with small active spaces. 

            // The python script to read and generate the file will need to be 
            // incorporated/rewritten as a part of DUCC.
            // XACC Ordering : OA | VA | OB | VB
            std::vector<int> XACC_order(2 * (nao + nav));

            // First nao values: first nao in occ_int_vec
            std::copy(occ_int_vec.begin(), occ_int_vec.begin() + nao, XACC_order.begin());

            // Next nav values: first nav in virt_int_vec
            std::copy(virt_int_vec.begin(), virt_int_vec.begin() + nav, XACC_order.begin() + nao);

            // Next nao values: last nao in occ_int_vec
            std::copy(occ_int_vec.end() - nao, occ_int_vec.end(), XACC_order.begin() + nao + nav);

            // Last nav values: last nav in virt_int_vec
            std::copy(virt_int_vec.end() - nav, virt_int_vec.end(), XACC_order.begin() + 2 * nao + nav);

            // DEBUG: Print XACC_order
            std::cout << "XACC_order: ";
            for (int idx : XACC_order) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;

            // // XACC Format:
            // for (int p = 0; p < nao + nav; ++p) {
            //     for(int q = 0; q < nao + nav; ++q) {
            //         h_xacc[p][q] = h[XACC_order[p]][XACC_order[q]];
            //         for(int r = 0; r < nao + nav; ++r) {
            //             for(int s = 0; s < nao + nav; ++s) {
            //                 v_xacc[p][q][r][s] = v[XACC_order[p]][XACC_order[q]][XACC_order[r]][XACC_order[s]];
            //                 //Have to check if this is correct (prefactor and order)
            //             }
            //         }
            //     }
            // }

            // CALL NWQSim with h_xacc, v_xacc, and scalar
        }

        // // STEP 4: Update the T1 and T2 tensors from NWQSim
        // for(const auto& combination : sorted_combinations) {
        //     for (int p = 0; p < nao + nav; ++p) {
        //         for(int q = 0; q < nao + nav; ++q) {
        //             h[XACC_order[p]][XACC_order[q]] = h_xacc[p][q];
        //             for(int r = 0; r < nao + nav; ++r) {
        //                 for(int s = 0; s < nao + nav; ++s) {
        //                     v[XACC_order[p]][XACC_order[q]][XACC_order[r]][XACC_order[s]] = v_xacc[p][q][r][s];
        //                     //Have to check if this is correct (prefactor and order)
        //                 }
        //             }
        //         }
        //     }
        // }
    }

        




    // When DUCC is called, the full T1 and T2 are taken in and 
    // zeroed according to the active space. Instead of copying 
    // the master then calling DUCC, DUCC can bemodified to create
    // copiesof T1 and T2 and then zero out those elements. 
}
