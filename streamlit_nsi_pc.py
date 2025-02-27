import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

def generate_consistent_pc_matrix(n, base_value=3):
    values = np.random.uniform(1, base_value, size=n)
    matrix = np.outer(values, 1/values)
    np.fill_diagonal(matrix, 1)
    return matrix

def apply_random_perturbation(matrix, deviation=0.1):
    n = matrix.shape[0]
    perturbed_matrix = matrix.copy()
    for i in range(n):
        for j in range(i + 1, n):
            perturbation = 1 + np.random.uniform(-deviation, deviation)
            perturbed_matrix[i, j] *= perturbation
            perturbed_matrix[j, i] = 1 / perturbed_matrix[i, j]
    return perturbed_matrix

def inconsistency_index_eigen(matrix):
    n = matrix.shape[0]
    eigenvalues = np.linalg.eigvals(matrix)
    lambda_max = np.max(np.real(eigenvalues))
    CI = (lambda_max - n) / (n - 1)
    return CI

def inconsistency_index_triads(matrix):
    n = matrix.shape[0]
    inconsistency = 0
    triad_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                expected = matrix[i, j] * matrix[j, k]
                actual = matrix[i, k]
                inconsistency += abs(expected - actual) / actual
                triad_count += 1
    return inconsistency / triad_count if triad_count > 0 else 0

def plot_histograms(values_list, labels, title):
    fig, ax = plt.subplots()
    for values, label in zip(values_list, labels):
        ax.hist(values, bins=10, alpha=0.5, label=label, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Wartość")
    ax.set_ylabel("Częstotliwość")
    ax.legend()
    st.pyplot(fig)

def monte_carlo_simulation(n, deviation, iterations=1000):
    ci_results = []
    triad_results = []
    
    for _ in range(iterations):
        consistent_matrix = generate_consistent_pc_matrix(n)
        nsi_matrix = apply_random_perturbation(consistent_matrix, deviation)
        
        ci_index = inconsistency_index_eigen(nsi_matrix)
        triad_index = inconsistency_index_triads(nsi_matrix)
        
        ci_results.append(ci_index)
        triad_results.append(triad_index)
    
    return ci_results, triad_results

def main():
    st.title("NSI PC Matrix Generator & Monte Carlo Simulation")
    
    matrix_sizes = st.multiselect("Wybierz rozmiary macierzy (n x n)", list(range(3, 21)), default=[5, 10, 15])
    deviations = st.multiselect("Wybierz poziomy perturbacji (D)", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], default=[0.05, 0.2, 0.4])
    iterations = st.number_input("Liczba iteracji Monte Carlo", min_value=100, max_value=1000000, value=1000, step=100)
    
    if st.button("Uruchom symulację Monte Carlo"):
        results = []
        labels = []
        
        for n in matrix_sizes:
            for deviation in deviations:
                ci_values, triad_values = monte_carlo_simulation(n, deviation, iterations)
                labels.append(f"n={n}, D={deviation}")
                
                st.subheader(f"Monte Carlo wyniki dla n={n}, D={deviation}")
                st.write(f"Średni Eigenvalue-based CI: {np.mean(ci_values):.4f}")
                st.write(f"Średni Triad-based Index: {np.mean(triad_values):.4f}")
                
                results.extend([(n, deviation, ci, triad) for ci, triad in zip(ci_values, triad_values)])
        
        df_results = pd.DataFrame(results, columns=["n", "D", "CI", "Triad Index"])
        
        # Eksport do CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Pobierz wyniki jako CSV", data=csv, file_name="monte_carlo_results.csv", mime="text/csv")
        
        # Eksport do Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name="Monte Carlo Results")
        excel_data = excel_buffer.getvalue()
        st.download_button("Pobierz wyniki jako Excel", data=excel_data, file_name="monte_carlo_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # Histogramy wyników Monte Carlo
        plot_histograms([df_results[(df_results["n"] == n) & (df_results["D"] == d)]["CI"] for n in matrix_sizes for d in deviations], labels, "Histogram CI dla różnych n i D (Monte Carlo)")
        plot_histograms([df_results[(df_results["n"] == n) & (df_results["D"] == d)]["Triad Index"] for n in matrix_sizes for d in deviations], labels, "Histogram Triad-based Index dla różnych n i D (Monte Carlo)")
    
if __name__ == "__main__":
    main()
