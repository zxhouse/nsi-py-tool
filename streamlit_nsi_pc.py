import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main():
    st.title("NSI PC Matrix Generator")
    
    n = st.slider("Rozmiar macierzy (n x n)", 3, 10, 5)
    deviations = st.multiselect("Wybierz poziomy perturbacji (D)", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.1, 0.3, 0.5])
    
    if st.button("Generuj NSI PC Matrices"):
        matrices = []
        labels = []
        
        for deviation in deviations:
            consistent_matrix = generate_consistent_pc_matrix(n)
            nsi_matrix = apply_random_perturbation(consistent_matrix, deviation)
            matrices.append(nsi_matrix.flatten())
            labels.append(f"D={deviation}")
            
            ci_index = inconsistency_index_eigen(nsi_matrix)
            triad_index = inconsistency_index_triads(nsi_matrix)
            
            st.subheader(f"Wskaźniki niespójności dla D={deviation}")
            st.write(f"Eigenvalue-based CI: {ci_index:.4f}")
            st.write(f"Triad-based Index: {triad_index:.4f}")
            st.write(pd.DataFrame(nsi_matrix))
        
        # Wyświetlanie histogramów dla różnych wartości D
        plot_histograms(matrices, labels, "Histogram wartości macierzy NSI dla różnych D")
    
if __name__ == "__main__":
    main()
