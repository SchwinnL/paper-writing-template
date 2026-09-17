# Example Background — Position Paper on Adversarial Alignment

Paper: "Position: We Need to Realign Incentives for Meaningful Progress in Adversarial Alignment for LLMs" (ICML 2026). Taxonomy section serving as background/framework.

---

\section{Adversarial robustness taxonomy}

To define differences between prior robustness research and emerging threat models in LLMs, we introduce a taxonomy inspired by established cybersecurity frameworks organized into goals and capabilities~\citep{papernot_practical_2017, carlini_evaluating_2019}. In the remainder of this work, we use this taxonomy to formally define differences between previous threat models and emerging ones in LLMs.

\subsection{Adversarial robustness goals}\label{sec:tax-goals}
The overarching goal of adversarial robustness research is to improve the robustness of neural networks. To this end, clearly defining this goal is crucial for consistent and meaningful comparisons of robustness evaluations across multiple works. 
This requires introducing a sound mathematical framework that specifies the conditions under which robustness should be achieved.
For example, considering evasion (test-time) robustness, the goal is to determine the worst-case change in the model's output under constrained changes in the input data:
\begin{equation}\label{eq:adv-goal-evasion}
    \delta^*(x) = \max_{\tilde{x} \in \gC(x)} \; d_{out} (f(x), f(\tilde{x})), 
\end{equation}
where $x$ is a fixed input, $f$ is the model, $d_{out}$ is a general distance metric between the model outputs, and $\gC(x)$ is the set of possible perturbations within a certain distance of $x$ (see \S\ref{sec:tax-capability}).

The optimization problem in \autoref{eq:adv-goal-evasion} establishes a fundamental definition of adversarial robustness in evasion settings, and serves as the foundation for its evaluation. Solving this optimization problem is NP-hard for most practical scenarios~\cite{katz2017reluplex}. To address this challenge, we require algorithms to assess robustness by computing lower and upper bounds on the model's robustness $\delta^*$. 
In the evasion example, lower bounds $\underline{\delta}(x)$ on $\delta^*(x)$ defined in \autoref{eq:adv-goal-evasion} can be derived using adversarial attacks, and upper bounds $\overline{\delta}(x)$ can be derived using robustness certification. Together, this allows to estimate the robustness of a given model: $\underline{\delta}(x) \leq \delta^*(x) \leq \overline{\delta}(x)$. In general, attacks and certificates provide (provable) bounds on robustness, and defenses (such as adversarial training) constitute strategies for achieving the goal of improved robustness.

\subsection{Capabilities}\label{sec:tax-capability}
We define capabilities of attackers by their knowledge, constraints, and computational overhead. 

\textbf{Knowledge.} Knowledge and access classifications typically include up to four categories, consisting of white-box, gray-box, black-box, and no-box ranging from full understanding and complete access to the model and its defenses (white-box) to no knowledge and no direct access (no-box), with varying degrees of knowledge and access in between (gray-box, black-box)~\citep{papernot_practical_2017, bose_adversarial_2020}.

The white-box threat model has emerged as the most common evaluation scenario in the adversarial robustness literature, as it enables the strongest attacks, thereby providing the most accurate quantification of the robustness of a respective defense~\citep{papernot_practical_2017}.
This aligns with Kerckhoff's Principle, which asserts that a system's security should not depend on obscurity, i.e., the secrecy of its design or implementation. Relying on obscurity introduces vulnerabilities, as once that obscurity is compromised, the entire system's robustness is at risk~\citep{sasa_kerk_2008,athalye_obfuscated_2018}.
The widespread adoption of the white-box threat model has been enabled through the open sourcing of newly published defenses and models.

\textbf{Constraints.} Constraints on adversarial perturbations serve two essential roles in robustness assessment: First, they reflect practical limitations, such as maintaining valid input domains (e.g., pixel values within image bounds) or ensuring malicious perturbations remain undetected. Second, they provide meaningful evaluation settings, as unconstrained adversaries can typically bypass any defense mechanism, making such scenarios more suitable as sanity checks than realistic threat models~\cite{goodfellow_explaining_2015}.
Formally, perturbation constraints $\gC$ are typically defined by bounding a distance metric between the original and perturbed inputs, $d_{in}(x, x') \leq \epsilon$. This formalization enables a consistent framework for evaluating and comparing robustness under defined conditions \cite{carlini_evaluating_2019}.

\textbf{Computational effort \& complexity.} We extend the attack capability definition to include practicality constraints on attacks and defenses, such as computational effort or pipeline complexity. Practicality constraints enable more realistic modeling of real-world attacks by reflecting the costs associated with both attacks and defenses. Cybersecurity threat models often assume an inverse relationship between these costs: a higher cost for an attacker, typically means a lower cost for the defender~\cite{barreno_security_2010}.
