# Example Introduction — Sampling in Adversarial Attacks

Paper: Integrating sampling into adversarial attack design for LLMs.

---

\section{Introduction}
\vspace{-0.1\baselineskip}
Large language models (LLMs) exhibit impressive performance across a wide range of tasks, yet ensuring their safe and reliable deployment continues to pose significant challenges~\citep{achiam2023gpt, schwinn2025adversarial}.
A fundamental goal of LLM-safety research is to minimize the risk of harmful behaviors or malicious exploitation~\citep{hendrycks2021unsolved,bai2022constitutional}.
Progress towards this goal is increasingly urgent given the magnitude of industrial deployments, where even low probabilities of harmful outputs can have severe real-world consequences \citep{jones2025forecasting}.

\vspace{-0.325\baselineskip}
While real-world risk is driven by large numbers of users sampling model completions at scale, current adversarial attacks against LLMs overlook this by largely relying on point estimates to evaluate attack success rates, usually based on a single, greedily generated response \citep{zou2023universal,zhu2023autodan,wang2024attngcg,chao2023jailbreaking}.

\vspace{-0.325\baselineskip}
To address this limitation of existing attacks, we propose integrating the process of sampling model completions directly into adversarial attacks against LLMs.
A key insight is that high-risk samples can be elicited with reasonable probability early in the optimization process.
By adopting more flexible sampling schedules that sample multiple completions throughout the attack, we substantially improve both the effectiveness and efficiency of existing attacks (\autoref{fig:hero}).

\input{figures/n-hero.tex}

\vspace{-0.325\baselineskip}
We show that in this setting, adversarial attacks can be naturally framed as a resource allocation problem under fixed compute budgets, where attackers must balance optimizing adversarial inputs to increase their likelihood of provoking a harmful response and sampling from the evolving output distribution.
Vulnerable models may produce harmful outputs with minimal optimization, whereas more robust models require extensive optimization before sampling becomes worthwhile.

\vspace{-0.325\baselineskip}
Our perspective also opens new directions for attack design. We illustrate this by developing a proof-of-concept adversarial objective which exploits sampling and does not require access to prompt- or model-specific affirmative response targets, making it model-agnostic and unbiased.

\vspace{-0.325\baselineskip}
In a thorough experimental evaluation, we demonstrate that because sampling is neglected as an attack vector, state-of-the-art attacks consistently overestimate LLM robustness.
Notably, our proposed perspective shift enables attack strategies that consistently outperform existing attacks in efficiency and attack success rate.
Moreover, we show that model safety rankings can differ when sampling is considered, indicating that greedy evaluation alone is insufficient for reliable safety assessment.


Our key contributions are:

\begin{itemize}[noitemsep,nolistsep,topsep=-2pt,leftmargin=0.6cm]
    \item We introduce a \textbf{sampling-aware framework for adversarial attacks} that treats sampling as a fundamental component of attack design, enabling principled resource allocation between optimization and sampling under fixed compute budgets.
    \item We demonstrate that sampling-awareness leads to \textbf{more efficient and effective adversarial attacks}, achieving up to two orders of magnitude reduction in computational cost and a 37 percentage point increase in attack success rates compared to state-of-the-art methods.
    \item We explain the efficiency gains of sampling by \textbf{investigating the impact of different optimization strategies} on the distribution of output harmfulness of the attacked model.
    \item We propose a novel \textbf{label-free and model-agnostic attack objective} based on maximizing the entropy of the distribution of the first predicted token, which is specifically designed to take advantage of sampling and leads to more natural responses.
\end{itemize}

Overall, our novel perspective provides a principled and efficient framework for adversarial attacks, enabling better risk assessment that could be used to evaluate future mitigation strategies.
