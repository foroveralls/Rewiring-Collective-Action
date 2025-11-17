# Supplementary Information: Parameter Tables

## Table S1: Network Typology Parameters (NetIn Package)

### PATCH Network Model Parameters

```latex
\begin{table}[h]
\centering
\caption{Parameters for PATCH (Preferential Attachment with Triadic Closure and Homophily) network generation using the NetIn package.}
\label{tab:patch_params}
\begin{tabular}{llp{7cm}}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
$n$ & 100--1089 & Network size (number of nodes) \\
$k$ & 8 & Average degree (number of initial connections per node) \\
$f_m$ & 0.5 & Minority fraction (proportion of minority group nodes) \\
$h_{MM}$ & 0.5 & Homophily for majority-majority connections \\
$h_{mm}$ & 0.5 & Homophily for minority-minority connections \\
$t_c$ & 0.5 & Clustering coefficient target \\
seed & 42 & Random seed for reproducibility \\
\hline
\end{tabular}
\end{table}
```

### DPAH Network Model Parameters

```latex
\begin{table}[h]
\centering
\caption{Parameters for DPAH (Duplication-Divergence with Preferential Attachment and Homophily) network generation using the NetIn package.}
\label{tab:dpah_params}
\begin{tabular}{llp{7cm}}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
$n$ & 100--1089 & Network size (number of nodes) \\
$f_m$ & 0.5 & Minority fraction (proportion of minority group nodes) \\
$d$ & 0.02 & Divergence probability (probability of deleting edges after duplication) \\
$h_{MM}$ & 0.5 & Homophily for majority-majority connections \\
$h_{mm}$ & 0.5 & Homophily for minority-minority connections \\
$\text{plo}_M$ & 2.0 & Power-law exponent for majority group degree distribution \\
$\text{plo}_m$ & 2.0 & Power-law exponent for minority group degree distribution \\
seed & 42 & Random seed for reproducibility \\
\hline
\end{tabular}
\end{table}
```

---

## Table S2: Node2Vec (N2V) Algorithm Parameters

```latex
\begin{table}[h]
\centering
\caption{Parameters for the Node2Vec algorithm used for network embedding-based rewiring. Node2Vec generates low-dimensional feature representations of nodes by performing random walks on the network.}
\label{tab:node2vec_params}
\begin{tabular}{llp{7cm}}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
dimensions & 64 & Dimensionality of the embedding space (size of feature vectors) \\
walk\_length & 40 & Length of each random walk (number of steps) \\
num\_walks & 5 & Number of random walks per node \\
context\_size & 10 & Context window size for Skip-Gram model \\
num\_threads & 1--2 & Number of parallel threads (1 for partial retrain, 2 for full retrain) \\
retrain\_threshold & 0.5 & Proportion of affected nodes that triggers full retraining ($>50\%$ of nodes) \\
\hline
\end{tabular}
\end{table}
```

---

## Table S3: WTF (Who-To-Follow) Algorithm Parameters

```latex
\begin{table}[h]
\centering
\caption{Parameters for the WTF (Who-To-Follow) recommendation algorithm. The algorithm combines personalized PageRank and SALSA (Stochastic Approach for Link-Structure Analysis) to identify potential connection targets.}
\label{tab:wtf_params}
\begin{tabular}{llp{7cm}}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
\multicolumn{3}{l}{\textit{PageRank Parameters}} \\
\hline
$\alpha$ & 0.70 & Damping factor (probability of continuing random walk vs. teleporting) \\
max\_iter & 100 & Maximum iterations for PageRank convergence \\
\hline
\multicolumn{3}{l}{\textit{Recommendation Parameters}} \\
\hline
topk & 5 & Number of top recommendations to generate per node \\
wtf\_freq & 10 & Cache refresh frequency (number of interactions before updating recommendations) \\
\hline
\multicolumn{3}{l}{\textit{SALSA Parameters}} \\
\hline
max\_iter & 100 & Maximum iterations for eigenvector centrality computation \\
hub\_candidates & topk & Number of hub nodes considered from PageRank results \\
\hline
\end{tabular}
\end{table}
```

---

## Table S4: General Model Parameters

For completeness, key general model parameters referenced in network generation:

```latex
\begin{table}[h]
\centering
\caption{General model parameters used across all network types and simulations.}
\label{tab:general_params}
\begin{tabular}{llp{7cm}}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
\multicolumn{3}{l}{\textit{Network Structure}} \\
\hline
$\langle k \rangle$ & 8 & Target average degree across all network types \\
$N$ & 100--1089 & Network size (varies by model and analysis) \\
\hline
\multicolumn{3}{l}{\textit{Agent Parameters}} \\
\hline
stubbornness & 0.6 & Agent resistance to opinion change \\
politicalClimate & 0.05 & External influence parameter favoring cooperation \\
randomness & 0.10 & Stochasticity in agent decision-making \\
$\mu_f$ & 0.5 & Mean friendship weight (tie strength) \\
$\sigma_f$ & 0.19 & Standard deviation of friendship weights \\
skew & $-0.20$ & Initial opinion distribution skew \\
$\sigma_{\text{init}}$ & 0.15 & Standard deviation of initial opinions \\
\hline
\multicolumn{3}{l}{\textit{Rewiring Parameters}} \\
\hline
$p_{\text{break}}$ & 1.0 & Probability of breaking a link after establishing new connection \\
$p_{\text{establish}}$ & 0.5 & Probability of establishing a new link (random rewiring only) \\
\hline
\end{tabular}
\end{table}
```

---

## Notes for Manuscript Integration

When referencing these tables in your Methods section "Network Typology", you can use:

```latex
Networks based on these models are generated with the NetIn package in Python \citep{pynetin}
(see SI Tables~\ref{tab:patch_params} and \ref{tab:dpah_params} for parameters). For the
Node2Vec and WTF rewiring algorithms, see SI Tables~\ref{tab:node2vec_params} and
\ref{tab:wtf_params}, respectively.
```

### Key Citations to Include:

1. **NetIn Package**:
   - Lim, M. (2023). netin: A Python package for network generation. Journal of Open Source Software, 8(88), 5450.

2. **Node2Vec**:
   - Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864).

3. **WTF Algorithm**:
   - Gupta, P., Goel, A., Lin, J., Sharma, A., Wang, D., & Zadeh, R. (2013). WTF: The who to follow service at Twitter. In Proceedings of the 22nd international conference on World Wide Web (pp. 505-514).

4. **SALSA**:
   - Lempel, R., & Moran, S. (2000). The stochastic approach for link-structure analysis (SALSA) and the TKC effect. Computer Networks, 33(1-6), 387-401.
