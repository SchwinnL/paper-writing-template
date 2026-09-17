# Example Introduction — Sampling for Evaluations in LLMs

Paper: Probabilistic evaluation framework for LLM unlearning.

---

Large Language Models (LLMs) are widely employed across various applications, from chatbots to code generation, relying on outputs generated through \textbf{probabilistic} decoding methods such as beam-search and multinominal sampling. Despite their probabilistic deployment, performance evaluations in LLMs predominately rely on \textbf{deterministic} point estimates, where outputs are generated through greedy
decoding. This raises a critical research question:

\begin{center}
\textit{Are deterministic evaluations adequate for assessing sensitive applications}\
\textit{or do they fall short in capturing the risks associated with probabilistic outputs?}
\end{center}

Current deterministic evaluation might result in a potential misalignment between evaluation and practical usage overlooking the inherent variability in model outputs. As a result, they could fail to account for both utility and potential risks associated with the model's entire output distribution. Yet, use cases like model alignment and unlearning demand precise model evaluations to mitigate the risk of harmful usage or privacy non-compliance during deployment. As illustrated in Figure~\ref{fig:figure1}, an unlearning algorithm may appear to successfully delete information in a deterministic setting yet still leak that information with a certain probability when outputs are sampled.
In many scenarios, leakage in even a small fraction of samples -- such as revealing a social security number, user passwords, or copyrighted information -- can be as problematic as leakage in every response, making deterministic evaluations insufficient to capture practical risks.

To address this, we evaluate the sufficiency of deterministic methods in an unlearning case study, focusing on whether they accurately reflect risks of information leakage in real-world~probabilistic~settings. We find that deterministic evaluations are insufficient, introduce a probabilistic view on unlearning and propose to evaluate the LLM's entire \textit{output distribution} instead of point estimates.
\newpage
Our main contributions are:

\begin{itemize}[noitemsep,nolistsep,topsep=-2pt,leftmargin=0.6cm]
    \item We demonstrate that simple multinominal sampling breaks all state-of-the-art unlearning algorithms that we evaluated in our experiments, retrieving most if not all of the unlearned information. We are the first to formally model the evaluation of LLMs from a novel probabilistic perspective and thereby capture the practical risk of information leakage more accurately than existing approaches.
    \item We propose a probabilistic evaluation framework consisting of a suite of principled metrics for comparing LLM output distributions with high-probability guarantees.
    \item A novel unlearning-loss based on entropy minimization and adaptive temperature scaling, significantly improving forget quality in probabilistic settings.
\end{itemize}

\begin{figure}[t!]
    \centering
    \input{figures/fig1.tex}
    \caption{We propose a novel \textbf{probabilistic evaluation framework} as a more reliable method for assessing LLMs capabilities. Existing evaluations are deterministic and rely on greedy decoding, where the most likely token is selected at each step, producing only a single output per query. Since \textit{in most practical applications LLMs generate outputs probabilistically}, previous evaluation schemes are insufficient: they overlook potential information leaks and falsely suggest successful unlearning.
    In contrast, in our probabilistic evaluation framework we directly consider the LLM's output distribution by sampling from the token probability distribution at each step to generate multiple sequences. In an empirical study, we show that all state-of-the-art unlearning methods leak information under our probabilistic setting, demonstrating that current deterministic evaluations are~insufficient.}
    \label{fig:figure1}
\end{figure}
