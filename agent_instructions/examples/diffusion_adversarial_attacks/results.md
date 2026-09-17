# Example Results — Diffusion Adversarial Attacks

Paper: Using diffusion LLMs for adversarial prompt generation. Transfer attack experiments on JailbreakBench.

---

\section{Experiments}

\begin{table*}
\centering
\caption{Attack success rate (ASR) comparison across open-source LLMs. Although \method{} (like BoN) is used as a transfer attack rather than being optimized for each model, it achieves the highest ASR across all models.}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{>{\centering\arraybackslash}m{2.5cm}
                >{\centering\arraybackslash}m{1.0cm}
                >{\centering\arraybackslash}m{1.5cm}
                >{\centering\arraybackslash}m{1.3cm}
                >{\centering\arraybackslash}m{1.8cm}
                >{\centering\arraybackslash}m{1.8cm}
                >{\centering\arraybackslash}m{1.2cm}}
\toprule
\textbf{Attack} & \textbf{Phi 4 Mini} & \textbf{Qwen 2.5 7B} & \textbf{Llama 3 8B} & \textbf{LAT Llama 3 8B} & \textbf{CB Llama 3 8B} & \textbf{Gemma 3 1B} \\
\hline
PGD & 38.0\% & 73.0\% & 11.0\% & 3.0\% & 3.0\% & 34.0\% \\
AutoDAN & 15.0\% & \textbf{100.0\%} & 60.0\% & 2.0\% & 2.0\% & 98.0\% \\
BoN & \textbf{100.0\%} & \textbf{100.0\%} & \textbf{100.0\%} & 3.0\% & 78.0\% & \textbf{99.0\%} \\
GCG & 98.0\% & 98.0\% & 71.0\% & 20.0\% & 7.0\% & 96.0\% \\
PAIR & 31.0\% & 62.0\% & 28.0\% & 15.0\% & 7.0\% & 64.0\% \\
Ours& \textbf{100.0\%} & \textbf{100.0\%} & \textbf{100.0\%} & \textbf{91.0\%} & \textbf{93.0\%} & \textbf{99.0\%} \\
\underline{99.0\%} & \underline{99.0\%} & \textbf{100.0\%} & \underline{85.0\%} & \underline{91.0\%} & \underline{98.0\%} \\
\bottomrule
\end{tabular}
\label{tab:results}
\end{table*}

The main goal of our experiments is to test whether generative models that learn the joint distribution $q(\vx, \vy)$ can act as natural adversarial attack generators. In particular, we evaluate whether \glspl{dllm} can produce adversarial prompts directly through their standard inference process. To answer this question, we conduct a series of experiments designed to assess
(I) the effectiveness and transferability of \gls{dllm}-based adversarial attacks across both open-source and proprietary models (\cref{sec:transfer}),
(II) whether standard inference in diffusion models conditioned on malicious targets increases harmfulness for target models as diffusion steps progress (\cref{sec:steps}),
(III) if guided sampling using signal from autoregressive models can further improve attack success (\cref{sec:guidance}), and
(IV) the linguistic quality of prompts generated via DLLMs, evaluated in terms of perplexity (\cref{sec:perplexity}).

\subsection{Setup}

\textbf{Models.} We attack $6$ different open-source models.  Specifically, we perform experiments on: Phi-4-Mini~\cite{abouelenin2025phi}, Qwen-2.5-7B~\cite{yang2025qwen3}, Gemma-3-1B~\citep{team2025gemma}, and Llama-3-8B-Instruct~\citep{grattafiori2024llama}. Moreover, we evaluate $2$ Llama-3-8B-Instruct models specifically finetuned for robustness using Circuit Breakers~\citep{zou2024improving} and latent adversarial training~\cite{sheshadri2024latent}.
For the \gls{dllm}, we use LLaDA-8B without instruction tuning~\cite{nie2025largelanguagediffusionmodels}, which was one of the first open-sourced \gls{dllm}s. We additionally attack ChatGPT-5 through the OpenAI API and set the model to the minimum thinking budget.
\newline
\textbf{Attacks.} We compare our method to six attack methods: GCG \citep{zou2023universal}, a variant of PGD \citep{geisler2024attacking}, AutoDAN \citep{liu2023autodan}, PAIR \citep{chao2023jailbreaking}, and Best-of-N (BoN)~\cite{hughes2024best}. These methods are selected for their strong performance in terms of attack success rate and efficiency. We generally use the original hyperparameter and evaluation setups as described in their respective papers. More details are provided in Appendix~\ref{sec:appendix-reproducibility}. For the proposed \method{} attack, we perform $75$ diffusion steps and perform $2000$ independent random restarts per behavior in the dataset.

\textbf{Benchmarks \& Metrics.} We conduct all experiments on the JailbreakBench (JBB) dataset, which contains 100 harmful behavior prompts~\cite{chao2024jailbreakbench}. Following prior work~\cite{mazeika2024harmbench}, we assess the harmfulness of model outputs using a judge LLM. Specifically, we use the fine-tuned StrongREJECT judge~\cite{souly2024strongreject}, which assigns a harmfulness score $\mathcal{H} \in [0,1]$. Outputs with $\mathcal{H} > 0.5$ are considered harmful. For attacks that sample multiple generations, we report the worst-case outcome~\cite{scholten2024probabilistic,beyer2025sampling}; if any sampled output is harmful, we count the model as broken and calculate the attack success rate (ASR) as the fraction of prompts for which the model is broken.

\subsection{DLLMs Yield Efficient and Strong Transfer Attacks against Black-box Models}\label{sec:transfer}

\textbf{Attack Success Rate.} Table~\ref{tab:results} reports attack success rates (ASR) for the evaluated adversarial methods on open-source LLMs. GCG, AutoDAN, PAIR, and PGD are white-box attacks and are optimized directly against each target model. BoN is model-agnostic and not optimized per model, relying on random character-level input perturbations. Finally, the proposed \method{} is used as a transfer attack, where adversarial inputs are generated through conditional generation with LLaDA-8B and then applied to the other models. Although we do not conduct any model-specific optimizations with the \method{} approach, it achieves the highest ASR across all models. BoN achieves similar ASR in most settings. However, it is not able to break the Latent Adversarial Trained (LAT) model. Specifically against the robustly fine-tuned Circuit Breakers and LAT models, \method{} considerably outperforms previous attack algorithms.

\textbf{Attack Efficiency.}
Next, we analyze the computational efficiency of the evaluated attack methods. Following the approach in~\cite{boreiko2024interpretable, beyer2025sampling}, we estimate the FLOPs required to reach a given ASR based on model size and number of tokens. For a fairer comparison between transfer and direct attacks, we consider only the total cost of computing the attack itself across all models, while omitting inference costs of the target model. While BoN generates perturbations essentially for free, its effectiveness is limited, particularly against robust models such as LAT. We compare the effectiveness of BoN perturbations and \method{} in Appendix~\ref{sec:appendix-attack-efficiency-sample}. Among the white-box methods, GCG and PGD exhibit comparable efficiency. PAIR performs similarly but tends to be slightly better against more robust models. AutoDAN is generally the least efficient, except on Gemma-1B where it achieves high ASR. However, this effectiveness mainly stems from its manually crafted human initialization, which has been noted by previous work~\cite{beyer2025llm}, and the subsequent optimization provides only a modest ASR improvement. In contrast, \method{} is substantially more efficient than all competing approaches. It is Pareto-optimal for most models, achieving the highest ASR within a given compute budget, and only underperforms AutoDAN on Gemma-1B. Against the more robust models, such as Circuit Breakers and LAT, \method{} remains efficient while achieving the highest ASR. Our results demonstrate the effectiveness of bypassing model-specific optimization and instead generating adversarial attacks directly through conditional generation.

\begin{figure*}[ht]
    \centering
    \includegraphics[]{plots/asr_vs_flops.pdf}
    \caption{Efficiency comparison between state-of-the-art LLM attacks and the proposed \method{}, which achieves near–Pareto-optimal performance in both attack success and generation cost for most models, particularly the robustly trained LAT and Circuit Breakers models.}
    \label{fig:asr_vs_flops}
\end{figure*}

\textbf{Transfer to Proprietary \glspl{llm}.}
To evaluate whether transfer attacks constructed with smaller \glspl{dllm} can also compromise proprietary systems, we applied the generated adversarial prompts to ChatGPT-5 via the OpenAI API. We follow the setting from the previous section but generate only 100 attacks per behavior with each method. \method{} yields by far the highest ASR ($53\%$), followed by BoN ($13\%$), GCG ($4\%$), and PGD ($1\%)$. For GCG and PGD, transfer attacks are computed on Llama-8B-Instruct. The results show that adversarial inputs generated through conditional generation can effectively transfer to state-of-the-art proprietary models. These findings highlight the practical risk posed by even small \glspl{dllm} as capable adversarial generators. Moreover, they emphasize the need to consider transfer-based threats from \glspl{dllm} when defending large proprietary \glspl{llm} in the future.

\subsection{DLLMs Optimize Harmfulness in Autoregressive Models}\label{sec:steps}

We now examine whether the diffusion-based forward process directly improves the surrogate objective introduced in Section~\ref{sec:armotized_search}.
To this end, we measure the harmfulness of responses generated by the autoregressive LLaMA model when conditioned on prompts sampled at different diffusion steps.
Figure~\ref{fig:judgescore_diffusionstep} illustrates how the harmfulness of the predicted prompt $\vx$ at each diffusion step
evolves over the course of denoising with the \gls{dllm}.
We observe a steady increase in judged harmfulness with the number of diffusion steps.
Since this evaluation is based on generations from the autoregressive model, the improvement indicates that prompts sampled from the surrogate distribution increasingly elicit harmful responses under the target model. This suggests that the \gls{dllm} indeed 1) optimizes the surrogate objective through conditional sampling alone, and 2) that the surrogate and target distributions exhibit a low fidelity gap in practice, as improvements in the surrogate space directly translate to increased harmfulness under the target model. Moreover, our results suggest that stronger diffusion models, and more generally any model that better captures the joint data distribution, are likely to yield even more effective attacks. These results provide empirical support for the assumptions in Section~\ref{sec:armotized_search}, confirming that the \gls{dllm} behaves as an effective amortized optimizer of the surrogate objective.

\begin{figure}[H]
    \centering
    \includegraphics{plots/x0_judge_score_vs_step.pdf}
    \vspace{-25pt}
    \caption{Average judged harmfulness of successful attacks increases smoothly over diffusion steps.}
    \label{fig:judgescore_diffusionstep}
\end{figure}

\subsection{Guidance Further Improves Attack Success}\label{sec:guidance}
\begin{figure}[H]
    \centering
    \includegraphics{plots/guidance_samples_small.pdf}
    \vspace{-24pt}
    \caption{Likelihood guidance improves ASR.}
    \label{fig:guidance}
\end{figure}
We now examine whether incorporating information from the target model can further reduce the gap between the surrogate and target distributions and improve attack performance. To this end, we guide the diffusion process with feedback from the target model, biasing generation toward prompts with higher target likelihood (see likelihood guidance in Section~\ref{sec:guided-conditional-sampling}). Note that we increased the number of diffusion steps to 100 to improve guidance. Figure~\ref{fig:guidance} shows that guided sampling consistently increases the attack success rate per generation. Guidance introduces additional computational overhead, as the target model must evaluate multiple candidate generations at each diffusion step. Despite this, we observe substantial ASR improvements per sample against highly robust models such as LAT and Circuit Breakers. Overall, the findings indicate that target-guided sampling can enhance attack performance beyond what is achievable with the standard diffusion process alone.

\subsection{DLLMs are Natural Low Perplexity Attackers}\label{sec:perplexity}

\begin{figure}[H]
    \centering
    \includegraphics{plots/perplexity_plot_box.pdf}
    \caption{Adversarial attacks generated using \method{}, conditioned on targets from JBB, exhibit similar perplexity compared to benign prompts from the UltraChat dataset~\cite{ding2023enhancing}, and harmful behaviors from JBB~\cite{chao2024jailbreakbench}, illustrating that conditioned generation leads to natural jailbreaks.}
    \label{fig:perplexity}
\end{figure}

To better understand the characteristics of prompts generated by \glspl{dllm}, we analyze their perplexity under the target autoregressive model. Perplexity is an uncertainty metric based on the likelihood assigned by the target model and is commonly used in filtering-based defenses~\cite{jain2023baseline}. Figure~\ref{fig:perplexity} shows that adversarial prompts sampled from the \gls{dllm} exhibit low perplexity comparable to benign prompts from the UltraChat dataset~\cite{ding2023enhancing} and to the original harmful behavior prompts in the JBB dataset. This indicates that the generated attacks remain semantically meaningful and are unlikely to be detected by simple likelihood-based defenses. Example generations are provided in Appendix~\ref{sec:appendix-examples}.

\subsection{Practical Considerations}

A few design decisions and implementation details that influenced our results are summarized below.

\textbf{Model Choice.}
We observe that the instruction fine-tuned version of LLaDA specialized for question answering, where the model is explicitly trained to predict $q(\vy \mid \vx)$, performs poorly as an adversarial generator. The finetuning appears to remove the ability of the model to invert the conditional and produce likely $\vx$ given $\vy$, thereby violating the surrogate fidelity assumption (cf. \cref{sec:armotized_search}) required for effective transfer between the \gls{dllm} and the autoregressive target models.

\textbf{Conditional Generation.}
Another natural choice for conditional generation would be to additionally constrain the generation of $\vx$ with a fixed prefix (e.g., $\text{prefix} \oplus \text{mask} \oplus \text{suffix}$, where $\oplus$ is a concatenation operator). Prefix conditioning could provide semantic guidance that simplifies the generation process. However, such conditioning would modify the original objective $\arg\max_{\vx} p_\theta(\vx \mid \vy)$ to a constraint form $\arg\max_{\vx_{k+1:T}} p_\theta(\vx_{k+1:T} \mid \vx_{1:k}=\vx^p, \vy)$, where $\vx^p$ denotes a fixed prefix of length $k$. This constraint limits the search space to prompts consistent with $\vx^p$ that still elicit a harmful response $\vy$. In practice, this restriction reduces sample diversity, making the optimization problem unnecessarily harder. We empirically find that prefix conditioning reduces attack success, for example, by making the model produce refusals immediately after the prefix.

\textbf{Masking the Target.}
During conditional generation, we mask the conditioning target $\vy$ in the diffusion process to remain consistent with the model's training distribution, where random tokens are progressively demasked in an unstructured manner.
Leaving $\vy$ unmasked would introduce a distribution shift.
Empirically, stochastic masking of the condition improves ASR in our experiments.

\textbf{Vocabulary Filtering.}
We remove special or system tokens from the vocabulary during generation. Allowing these tokens leads the model to insert surrogate-specific chat template tokens, resulting in non-transferable jailbreaks. We also observe a general drop in attack success, even on the surrogate model, when these tokens are included.
