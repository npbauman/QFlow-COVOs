#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric> // For accumulate
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
    int nao   = 3; // Number of active occupied orbitals
    int nav   = 3; // Number of active virtual orbitals
    int nocc  = 10; // Total number of occupied orbitals
    int nvirt = 20; // Total number of virtual orbitals

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


    // STEP 3: 
    

        
    // }




    // {"occ_int", {range(occ_alpha_ext, n_occ_alpha), range(n_occ_alpha + occ_beta_ext, nocc)}},
    // {"virt_int",
    //  {range(nocc, nocc + vir_alpha_int),
    //   range(nocc + n_vir_alpha, nocc + n_vir_alpha + vir_beta_int)}},


    // When DUCC is called, the full T1 and T2 are taken in and 
    // zeroed according to the active space. Instead of copying 
    // the master then calling DUCC, DUCC can bemodified to create
    // copiesof T1 and T2 and then zero out those elements. 
}
