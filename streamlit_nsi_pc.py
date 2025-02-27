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

def plot_histogram(values, title):
    fig, ax = plt.subplots()
    ax.hist(values, bins=10, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Wartość")
    ax.set_ylabel("Częstotliwość")
    st.pyplot(fig)

def main():
    st.title("NSI PC Matrix Generator")
    
    n = st.slider("Rozmiar macierzy (n x n)", 3, 10, 5)
    deviation = st.slider("Poziom perturbacji (D)", 0.0, 0.5, 0.1, step=0.05)
    
    option = st.radio("Wybierz metodę wprowadzania danych:", ("Losowa macierz", "Ręczne wprowadzenie"))
    
    if option == "Losowa macierz":
        if st.button("Generuj NSI PC Matrix"):
            consistent_matrix = generate_consistent_pc_matrix(n)
            nsi_matrix = apply_random_perturbation(consistent_matrix, deviation)
            
            st.subheader("NSI PC Matrix")
            st.write(pd.DataFrame(nsi_matrix))
            
            ci_index = inconsistency_index_eigen(nsi_matrix)
            triad_index = inconsistency_index_triads(nsi_matrix)
            
            st.subheader("Wskaźniki niespójności")
            st.write(f"Eigenvalue-based CI: {ci_index:.4f}")
            st.write(f"Triad-based Index: {triad_index:.4f}")
            
            # Histogram wyników
            plot_histogram(nsi_matrix.flatten(), "Histogram wartości macierzy NSI")
    
    elif option == "Ręczne wprowadzenie":
        st.subheader("Wprowadź macierz PC")
        manual_matrix = []
        for i in range(n):
            row = st.text_input(f"Wiersz {i+1} (wartości oddzielone spacją)", " ".join(["1.0"] * n))
            manual_matrix.append([float(x) for x in row.split()])
        manual_matrix = np.array(manual_matrix)
        
        if st.button("Oblicz wskaźniki"):
            ci_index = inconsistency_index_eigen(manual_matrix)
            triad_index = inconsistency_index_triads(manual_matrix)
            
            st.subheader("Wskaźniki niespójności")
            st.write(f"Eigenvalue-based CI: {ci_index:.4f}")
            st.write(f"Triad-based Index: {triad_index:.4f}")
            st.subheader("Twoja macierz")
            st.write(pd.DataFrame(manual_matrix))
            
            # Histogram wyników
            plot_histogram(manual_matrix.flatten(), "Histogram wartości wprowadzonej macierzy")
        
if __name__ == "__main__":
    main()
