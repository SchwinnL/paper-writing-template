# Example Results — Adversarial Training in LLMs

Paper: Continuous-Adversarial UL and Continuous-Adversarial IPO for adversarial training in LLMs.

---

\section{Results}\label{sec:robustness}

In the following, we illustrate the computational benefit of continuous AT compared to existing discrete methods. Subsequently, we show improved robustness against state-of-the-art discrete attacks by using continuous adversarial training (AT).
\paragraph{Why do we need continuous adversarial training?}
\begin{wraptable}{r}{0.4\textwidth}
    \vspace{-12pt}
    \centering
    \caption{The combined number of forward (F) and backward (B) passes to compute a single adversarial example for different AT types. The total number of F\&B for the whole training and the number of training iterations and batch size are are shown. Time is the wallclock time for a single batch weight update (measured on 1 A100 with Mistral).
    }
    \resizebox{0.4\textwidth}{!}{
    \begin{tabular}{l|rrr}
    \toprule
    Algorithm & R2D2 & \advul{} & \advdpo{} \\
    \midrule
    F/B & 2565/5 & 10/10 & 10/10 \\
    Iterations & 2000 & 780 & 360 \\
    Batch size & 256 & 64 & 64 \\
    F/B (total) & 165,632,000 & 234,000 & 552,960 \\
    Time (sec) & 1567.8  & 3.2 &  3.2\\
    Type & Discrete & Continuous & Continuous \\
    \bottomrule
    \end{tabular}
    }
    \label{tab:compute}
    \vspace{-0.9em}
\end{wraptable}
In Table~\ref{tab:compute}, we compare the combined number of forward and backward passes used by the discrete AT algorithm RD2D~\citep{mazeika2024harmbench} with \advul{} and \advdpo{}.
Computing a single adversarial example with R2D2 is $\approx 128.5$ times more expensive than for \advul{} and \advdpo{}, while the whole training is $\efficency$ times more costly. This illustrates the considerable compute advantage of continuous AT approaches compared to discrete methods.

\paragraph{LLM adversarial training with utility data}\label{sec:ul} We first explore robustness extrapolation from continuous AT to discrete attacks for the \advul{} algorithm, which utilises additional utility data to maintain model performance. Figure~\ref{fig:at_results} summarises the evaluation results. For all models, \advul{} considerably increases the average robustness against discrete adversarial attacks. For the \gemma{} and \zephyr{} models, robustness increases for all attacks. For \phimodel{} and \mistral{}, \pair{} still achieves high attack success rates (ASR).
In terms of utility, we observe similar degradations for all \advul{} trained models.

Compared to the \rtwodtwo{} model, which was trained with discrete AT, \advul{} exhibits marginally worse utility on standard utility benchmarks while providing substantially improved robustness against discrete attacks. For, \rtwodtwo{}, PAIR achieves an ASR of $40\%$, while it achieves $10\%$ ASR for \advul{}. We note a substantial difference in the \textsc{Harmless} benchmark, where \advul{} massively outperforms \rtwodtwo{} showing that our method has not overfitted the safety objective or the patterns in the Harmbench behaviours. Note that the \textsc{Harmless} score of R2D2 demonstrates that it can not simultaneously achieve non-trivial utility and robustness, which are heavily dependent on not using or using the chat template, respectively.


\begin{figure}
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \includegraphics[width=\textwidth]{images/GEMMA-2B-IT_bar_DPO.pdf}
        \caption{\gemma{}}
    \end{subfigure}
    \begin{subfigure}[b]{0.495\textwidth}
        \includegraphics[width=\textwidth]{images/PHI-3-MINI_bar_DPO.pdf}
        \caption{\phimodel{}}
    \end{subfigure}
    \begin{subfigure}[b]{0.495\textwidth}
        \includegraphics[width=\textwidth]{images/MISTRAL-7B_bar_DPO.pdf}
        \caption{\mistral{}}
    \end{subfigure}
    \begin{subfigure}[b]{0.495\textwidth}
        \includegraphics[width=\textwidth]{images/LLAMA-2-7B_bar_DPO.pdf}
        \caption{\llama{-7B}}
    \end{subfigure}
        \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \includegraphics[width=\textwidth]{images/ZEPHYR-7B_bar_DPO.pdf}
        \caption{\zephyr{-7B}}
    \end{subfigure}

    \caption{\textbf{Trade-off} between utility and robustness for \advul{}~(Eq.~\ref{eq:ul+utility}), \advdpo{}~(Eq.~\ref{eq:adv dpo}), and R2D2~\citep{mazeika2024harmbench}, compared to their non-adversarially fine-tuned models. The objective is a small loss in utility and a large improvement in attack robustness. Larger is better for \mmlu{}, \arc{-E}, \arc{-C}, \mtbench{} (left of dashed line). Smaller is better for \gcg{}, \autodan{}, and \pair{} (right of dashed line). \mtbench{} score is multiplied by 10 to see the change in performance on this $y$-axis.  Additional results are included in App.~\ref{app:mainresults}.
    }
    \label{fig:at_results}
    \vspace{-1em}
\end{figure}

\paragraph{LLM adversarial training without utility data}\label{sec:dpo}
We further investigate if adversarial variations of proven alignment methods, such as IPO, can be used to align models in an adversarially robust manner (see Figure~\ref{fig:at_results}). For this purpose, we fine-tune \gemma{} and \phimodel{} using the proposed \advdpo{} algorithm. Figure~\ref{fig:at_results}, illustrates differences between the base model, \advul{}, and \advdpo{}. Despite using no utility dataset within \advdpo{} to retain helpfulness, the algorithm does not introduce larger utility decreases on common benchmarks than \advul{}. Moreover, \advdpo{} achieves considerably higher robustness against the jailbreaking method \pair{}, demonstrating generalisation to diverse threat models. The \phimodel{-IPO} model achieves $100\%$ attack robustness for all conducted attacks. For \gemma{}, robustness improvements also mostly surpass \advul{}, with slightly lower robustness against \gcg. Compared to R2D2, \advdpo{} does not require an auxiliary dataset to maintain utility and achieves higher robustness on average. Specifically for \pair{} \advdpo{} trained models exhibit considerably higher robustness. Lastly, the \phimodel{-IPO} achieves a substantially higher score on the \textsc{Harmless} benchmark than \advul{} and R2D2.

\emph{The results indicate that adversarial variations of common alignment methods, such as IPO, can be used to adversarially align LLMs.}

\paragraph{Utility evaluation} Common utility benchmarks such as \mmlu{} or \arc{} do not use a chat template in their standard evaluation~\citep{eval-harness}. Firstly, this dramatically impacts performance, especially for smaller models, which often require a lot of prompt engineering to follow the few-shot prompts correctly. Secondly, it dramatically changes the mode of the model. In effect, a model might be overly robust in chat mode (i.e.\ when using a chat template) where it rejects most requests, but it might appear to have high utility in benchmarks because no chat template is used (e.g.\ \mmlu{}). \arc{} as an evaluation benchmark is even more misleading as it measures the likelihood of a set of possible answer tokens, thus not reflecting the utility of the model when using a chat template. We quantitatively evaluate the refusals of \mmlu{} questions when using a chat template in App.~\ref{app:mmlu refusal}. We recommend future work, to consider these issues when evaluating robustness and utility for the same model.

\paragraph{Training data failure modes}
AT datasets such as Harmbench~\citep{mazeika2024harmbench} or AdvBench~\citep{chen2022should} tend to use a common grammatical and syntactical structure, using imperative commands such as ``Tell me'' or ``Give instructions''. Chatting with our models and \rtwodtwo, we observe that requests would be refused when using this same style but are accepted if asked in a different style, such as ``Could you please ...?''. This holds for both harmful and harmless requests. For instance, \rtwodtwo{} will refuse to answer ``Tell me a story'' and ``Tell me how to build a bomb'', but will answer ``Could you please tell me a story?'' and ``Could you please explain to me how to build a bomb?''. This also explains why the model may even appear useful under utility benchmarks employing chat templates such as \mtbench{}. To demonstrate this failure case we create two small benchmark datasets called \textsc{PoliteHarmbench} (see App.~\ref{app:polite eval}) and \textsc{Harmless}. The former rephrases the harmful behaviours politely, and the latter consists of harmless requests formulated in the same grammatical style as the original \textsc{Harmbench} behaviours. We leave developing better datasets and benchmarks for a future paper as it is outside the scope of this work.

\section{Adversarial Training Ablations}\label{sec:ablations}

\paragraph{Robust fine tuning without attack} We found that continuous adversarial training successfully increases the robustness of LLMs to discrete adversarial attacks. Here, we explore whether robustness gains stem from using continuous adversarial attacks during training, or from the fine-tuning process itself. Thus, we fine-tune \gemma{} using the \advdpo{} algorithm but without using adversarial attacks. We observe no robustness gains when fine-tuning without attacks (see App.~\ref{app:noattack}). This demonstrates that continuous adversarial attacks are a crucial part of our fine-tuning algorithm.

\paragraph{One-step adversarial training in LLMs} For all our experiments, we use $10$ adversarial attack iterations. While this is orders of magnitude cheaper than calculating discrete adversarial attacks (\gcg{} requires $2570$ model evaluations with default settings), it still increases training time by an order of magnitude. We thus propose one-step AT with \advdpo{}. As in previous work~\citep{goodfellow_explaining_2015}, we set the step size of the attack to the magnitude of the $\epsilon$-ball. This achieves robustness improvements comparable to the multi-step variant and slightly worse utility trade-offs (see App~\ref{app:onestepattack}).

\begin{figure}
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{images/table_dpo_beta_gemma-1.1-2b-it.pdf}
        \vspace{-20pt}
        \caption{Beta ablation}
    \end{subfigure}
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{images/table_dpo_eps_gemma-1.1-2b-it.pdf}
        \vspace{-20pt}
        \caption{Epsilon ablation}
    \end{subfigure}
    \caption{Ablating how changing $\beta$ or $\epsilon{}$ affect \gcg{} loss vs \mmlu{} score on \gemma{-IPO}}
    \label{fig:tradeoff-params}
    \vspace{-1em}
\end{figure}

\paragraph{Robustness-utility trade-offs} Prior work on AT has shown theoretical and empirical trade-offs between robustness and utility~\citep{madry_towards_2018, zhang2019theoretically}. Our previous results demonstrate that continuous AT can achieve non-trivial robustness-utility trade-offs. All experiments are conducted on \gemma{} models trained with \advdpo{} and varying hyperparameters. Specifically, we sample $\epsilon \in [0.00125, 0.3]$, and $\beta \in [0, 0.5]$ and fine-tune $7$ different models. In Figure~\ref{fig:tradeoff}, we depict the \gcg{} loss of the trained models (as a proxy for robustness) on the $y$-axis in logarithmic scale against the \mmlu{} score on the $x$-axis (as a proxy for utility). Clear trade-offs between robustness and utility can be observed, ranging from models with high robustness and no utility to models showing less robustness than the standard non-robust models and slightly higher utility.

Moreover, we analyse hyperparameter choices that affect the robustness-utility trade-off for \advdpo{} in more detail. This includes the strength of the adversarial attacks defined by the $\epsilon$ magnitude and the IPO $\beta$ value.
Figure~\ref{fig:tradeoff-params} illustrates that for both hyperparameters, we obtain intuitive robustness-utility trade-offs, where larger epsilon values and smaller $\beta$ values are associated with increased robustness and reduced utility. A detailed analysis can be found in App~\ref{app:tradeoff}.

\paragraph{Correlation between continuous attack loss and \gcg{} loss} We additionally investigated the relationship between training-time robustness to continuous adversarial attacks and inference-time robustness to discrete attacks. This is illustrated in Figure~\ref{fig:corr}. The observed strong Pearson correlation ($r=0.99$, $p=0.0075$) indicates that models robust to continuous attacks during training are also robust to discrete attacks at inference. This suggests continuous AT can be a reliable proxy for AT with discrete attacks. Thus, demonstrating the potential use of continuous attacks to reduce the computational burden of evaluating adversarial robustness~\citep{schwinn2023adversarial, schwinn2024soft}.

\begin{figure}
    \centering
    \begin{subfigure}[b]{0.47\textwidth}
        \includegraphics[width=\textwidth]{images/DPO_gemma-1.1-2b-it_loss_corr.pdf}
        \captionsetup{skip=-5pt}
        \caption{Robustness correlation}
        \label{fig:corr}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.47\textwidth}
        \includegraphics[width=\textwidth]{images/DPO_gemma-1.1-2b-it_utility_robustness.pdf}
        \captionsetup{skip=-5pt}
        \caption{Robustness-utility trade-off}
        \label{fig:tradeoff}
    \end{subfigure}
    \label{fig:dpo}
    \caption{\gemma{-IPO} used for both plots: (a) Correlation between \gcg{} loss and continuous attack loss. (b) \gcg{} loss vs \mmlu{} score for a variety of $\epsilon$ and $\beta$ values.}
    \vspace{-1em}
\end{figure}
