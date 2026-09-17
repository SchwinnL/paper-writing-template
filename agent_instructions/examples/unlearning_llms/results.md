# Example Results — Unlearning in LLMs

Paper: Partial model collapse (PMC) for machine unlearning. Experiments on TOFU dataset.

---

\begin{figure}[t]
    \vspace{-0.2cm}
    \centering
    \begin{minipage}{0.49\textwidth}
        \centering
        \input{figures/eval/phi-grid.pgf}
    \end{minipage}
    \hfill
    \begin{minipage}{0.49\textwidth}
        \centering
        \input{figures/eval/grid-llama.pgf}
    \end{minipage}%
    \vspace{-0.1cm}
    \caption{Partial model collapse (PMC) significantly dominates baselines and expands the Pareto-front w.r.t.\ utility and unlearn quality for (a) Phi-1.5 and (b) Llama-3.2-3B-Instruct. While existing methods (GA, GD, NPO, SimNPO, and IDK) also unlearn, they cannot deviate much from the model before unlearning without compromising the model's general capabilities. Orange lines indicate fine-tuned models before unlearning, max.\ unlearn quality is 2. Stars represent dominating points.}\label{fig:eval}
    \vspace{-0.4cm}
\end{figure}

\section{Experimental evaluation}\label{sec:exp}
\vspace{-0.15cm}
We experimentally demonstrate that the information loss in model collapse can be leveraged to achieve machine unlearning. We also identify negative side effects in existing unlearning methods that directly optimize on unlearning targets and showcase positive effects of our approach, such as robustness and reduced leakage under sampling. We provide additional results in~\autoref{app:add-results}, and refer to \autoref{app:expSetup} for experimental setups, implementation details and reproducibility instructions.

\textbf{Experimental setup.}  We perform experiments on the TOFU dataset \citep{maini2024tofu}, a fictitious dataset of 4,000 question-answering pairs designed for machine unlearning. We fine-tune models on the full dataset and perform unlearning on the ``forget10'' split, since it has the largest forget set and thus corresponds to the most challenging split in the dataset. We perform experiments for two models: Phi-1.5 \citep{li2023textbooks} since it is smaller and extensively studied in the unlearning literature, and Llama-3.2-3B-Instruct \citep{grattafiori2024llama} since it is a more recent model with strong performance across tasks. Experiments are performed on A100 and H100 GPUs.

\textbf{Baselines.} As baselines we consider \textit{Gradient Ascent} (GA) and \textit{Gradient Difference} (GD) \citep{liu2022continual}, as well as \textit{Negative Preference Optimization} (NPO) \citep{zhang2024negative} and its simplified version (SimNPO) \citep{fan2024simplicity}. We also introduce a new baseline that fine-tunes on retain and forget data but simply replaces all forget-answers with the phrase ``\textit{I don't know.}'' (IDK).

\textbf{Metrics}.  We evaluate using recall ROUGE-L scores \citep{lin2004rouge}, i.e.\ the longest common subsequence between the model's greedy output and the ground truth. Unlearning performance is quantified using the sum of ROUGE-L scores on the forget and paraphrased-forget sets---the latter is an additional TOFU dataset allowing to quantify generalization of unlearning. We report \textit{unlearn quality} as the maximal score minus the achieved score (such that larger is better), and \textit{utility}, measured as the sum of ROUGE-L scores on the retain set $D_r$ and two additional TOFU datasets: world facts (117 questions) and real authors (100 questions), which allow to assess general knowledge retention.

\textbf{Reward function.} The design of the reward function $r(x)$ is decisive for achieving the envisioned target after unlearning and highly application-dependent. Our goal in this paper is to remove the model output for forget questions and we therefore use the ROUGE-L score between the model's original and current output, i.e., $r(x) = 1-\text{ROUGE-L}(\hat{x}, y) \in [0,1]$, where $y$ is the model's original answer for forget question $q\in \gD_f$ and $\hat{x}$ is the sampled output as described in \cref{algo:alg3}.
\vspace{-0.2cm}

\subsection{Partial model collapse achieves more effective unlearning}
\vspace{-0.15cm}
In a series of experimental evaluations, we compare our proposed partial model collapse (PMC) to the baselines described above (GA, GD, NPO, SimNPO, and IDK). Since all methods involve multiple different hyperparameters, we perform a grid search for each method.

Specifically, we explore 100 different configurations for each method to ensure a fair comparison, covering a broad range of hyperparameter values while keeping the number of trials consistent across methods (see \autoref{app:expSetup} for exact search spaces for the grid search). We repeat the experiment for each configuration five times using different random seeds, and report mean utility and unlearn~quality.

Notably, PMC significantly dominates all baselines in the utility-unlearning trade-off and expands the Pareto-front, achieving strong unlearn quality while maintaining high utility across most hyperparameter configurations (\autoref{fig:eval}). In contrast, existing methods achieve lower unlearn quality and/or compromise the model's general capabilities.
We observe that PMC-unlearned Phi-1.5 models often answer with generic phrases like ``The answer is not available'' (or similar), while Llama-3.2-3B-Instruct models achieve almost optimal unlearn quality by refusing to answer exclusively for forget and paraphrased forget questions (despite performing unlearning only on the~former). Note that the desired response for forget questions depends on the use-case and can be adjusted by modifying the reward~function.

The underlying reason that our method achieves such strong~results without compromising the model's general capabilities is that we do not explicitly optimize on unlearning targets. Instead, we fine-tune on responses that are already likely under the model's own distribution. This way, we can force the model to diverge from the unlearning targets toward the optimal reward (guided by the reward function), rather than pushing the model away from explicit~targets.

\textbf{Extended utility analysis.} Although PMC is optimized only on the retain data to preserve utility, we find that its impact on overall model utility beyond the TOFU utility dataset is minimal in practice. Results on the ARC-Challenge, ARC-Easy, and MMLU benchmarks (\autoref{app:add-results}) show that PMC-unlearning has minimal to negligible effect on general model utility.
\vspace{-0.2cm}

\begin{figure}[t!]
    \centering
    \begin{minipage}{0.33\textwidth}
        \centering
        \input{figures/limitations/utility_deg.pgf}
    \end{minipage}
    \hfill
    \begin{minipage}{0.33\textwidth}
        \input{figures/limitations/mpc_least_correct.pgf}
        \centering
    \end{minipage}%
    \hfill
    \begin{minipage}{0.33\textwidth}
        \centering
        \input{figures/limitations/mpc_prob.pgf}
    \end{minipage}%
    \vspace{-0.1cm}
    \caption{Limitations of unlearning methods optimizing on unlearning targets: (a) Side effects on unrelated datasets. (b) Accuracy when selecting least likely answer across quantiles (black line is random guessing). (c) Distribution of minimum probabilities across all multiple-choice options.}\vspace{-0.5cm}
    \label{fig:existing_limitations}
\end{figure}

\vspace{-0.1cm}
\subsection{PMC overcomes limitations of methods optimizing on unlearning targets}
\vspace{-0.1cm}
Existing unlearning methods predominantly incorporate the unlearning target directly into their objectives. We argue that this approach may have subtle effects on model properties related to the unlearning targets, such as distorting token probabilities and leaking information about the private data used during unlearning optimization. Yet, the utility of unlearning models is typically evaluated using benchmark datasets or by comparing them to a retrained model~\citep{maini2024tofu}. As a result, existing evaluations may miss subtle changes in the generation properties of unlearned models.

\textbf{Generation capability on unrelated datasets.} First, we study generations of tokens targeted in the unlearning optimization and investigate whether existing methods compromise the model's ability to generate such tokens.
We argue that unlearning should prevent models from revealing unlearned information, but this effect must be limited to the unlearning context. It should not affect token generation in unrelated settings, as most tokens in forget sets are not semantically tied to the unlearning task but rather to sentence structure. For example, if we want to unlearn that John Doe is a carpenter, existing methods would minimize the probability of ``carpenter'' when asked about John Doe's profession. However, these methods should not reduce this probability in unrelated contexts.

To investigate such potential side effects, we compare the probability of generating tokens present in TOFU compared to the first $100$ text chunks of the wikitext-2-raw-v1 train split~\citep{merity2016pointer}.
\autoref{fig:existing_limitations} (a) shows the probability difference between unlearned models (NPO and our proposed \ac{method} method) and the base model: $p_{un}(x_t|x) - p_{base}(x_t|x)$, where $x_t$ is a token present in the forget set, $x$ is the context of this token in the wikitext dataset, and $p(x_t|x)$ it the probability of $x_t$ given the context. As the base model, we use a model fine-tuned exclusively on the retain set, with no exposure to the forget data. NPO substantially reduces the probability of generating forget set tokens also present in wikitext. A considerable number of tokens that originally gets assigned a high probability from the base model (e.g., close to $1$) get assigned a probability of $0$ from the unlearned model (indicated by $-1$ values in the figure).
In contrast, our method preserves generation probabilities, exhibiting token probabilities that are neither systematically increased nor decreased. For PMC, the differences follow a zero-mean Gaussian distribution with small variance, whereas they are skewed to the left for NPO ($-0.12$ mean). This shows that methods dependent on unlearning targets can considerably distort token probabilities even out-of-context of the unlearning task.

\textbf{Probability distribution in multiple-choice settings.}
Second, we hypothesize that existing unlearning methods exhibit information ``leakage'' by unnaturally reducing the probability of correct answers, potentially allowing adversaries to identify forgotten information by simply selecting the least likely option. To further investigate such potential negative side effects, we created a multiple-choice dataset from the TOFU forget10 set by converting a subset of $84$ questions into multiple-choice (MPC) format (Appendix~\ref{app:prompt_template}).  We use the inverse perplexity of every answer as its score and turn scores into probabilities by normalizing them. Moreover, for the correct answers in the MPCs we used rephrased versions of the correct TOFU answers rather than exact matches to demonstrate that leakage can occur even for semantically similar but non-identical formulations.

Our experiments provide clear empirical evidence for our leakage hypothesis. \autoref{fig:existing_limitations} (b) shows accuracy when selecting the least likely answer across quantiles ordered by minimum probability among choices. NPO exhibits high accuracy for questions where the minimum probability is very low, indicating that the correct answer frequently becomes the least likely option. Conversely, our method shows no such pattern. \autoref{fig:existing_limitations} (c) shows the distribution of minimum probabilities across all multiple-choice options. Here, NPO's distribution clusters near zero, further confirming that target-based unlearning unnaturally suppresses correct answer probabilities even in rephrased~contexts.

\begin{figure}[t!]
    \centering
    \begin{minipage}{0.34\textwidth}
        \centering
        \input{figures/eval/phi-ablation-num-epochs.pgf}
    \end{minipage}
    \hfill
    \begin{minipage}{0.32\textwidth}
        \centering
        \input{figures/eval/phi-ablation-num-samples.pgf}
    \end{minipage}%
    \hfill
    \begin{minipage}{0.32\textwidth}
        \centering
        \input{figures/eval/phi-ablation-lambda.pgf}
    \end{minipage}%
    \caption{Ablation studies on (a) number of epochs, (b) number of samples, and (c) trade-off parameter $\lambda$. Dashed line is the fine-tuned model before unlearning. Shadows/bars indicate standard~deviation.}
    \label{fig:ablations}
    \vspace{-0.2cm}
\end{figure}

\vspace{-0.3cm}
\subsection{PMC is more robust against sampling and prefilling attacks}\label{sec:sampling}
\begin{wrapfigure}{r}{0.3\textwidth}
\vspace{-0.35cm}
  \centering
  \input{figures/eval/sampling-attack.pgf}
  \caption{PMC is more robust against sampling and prefilling attacks. Lower average worst-case leakage is better.}\label{fig:sampling-attack}%
\end{wrapfigure}
Finally, we demonstrate that PMC exhibits substantially greater robustness against sampling and prefilling attacks compared to prior approaches. To evaluate robustness under sampling, we draw 100 answers from the output distribution of the unlearned model, and compute the ROUGE-L score between each sampled answer and the ground truth answer. We then compute the maximum (worst-case) ROUGE-L score per question and report the average across all forget questions (see \autoref{app:expSetup} for full experimental setup). The results in \autoref{fig:sampling-attack} show that PMC significantly reduces leakage under sampling, in stark contrast to existing methods. While the simple supervised fine-tuning IDK baseline also reduces leakage under sampling, this effect is largely superficial. To demonstrate this, we perform a prefilling attack in which the model is prompted with a forget question and forced to continue from the prefix ``The answer is:''.  This attack bypasses the fine-tuned response and reveals that the IDK baseline still encodes substantial information about the unlearned answers, leading to high leakage.
Notably, while existing methods exhibit considerable leakage, PMC is the first approach to achieve more robust unlearning across both attack settings.

\subsection{Ablations studies for partial model collapse machine unlearning}

We perform ablation studies on PMC's hyperparameters under Phi-1.5 (additional results in \autoref{app:add-results}, details in \autoref{app:expSetup}). First, the number of training epochs strongly influences unlearning performance: while baseline methods converge after 10 epochs, PMC continues to improve unlearn quality without significantly affecting utility even after 20 epochs (\autoref{fig:ablations}a). Second, increasing the number of samples also enhances unlearning, with utility remaining stable for the first six epochs; larger sample sizes show slightly higher variance in utility (\autoref{fig:ablations}b). Finally, we ablate the unlearn-utility trade-off parameter $\lambda$, observing that larger values improve utility but can degrade unlearn quality, highlighting the importance of selecting $\lambda$ to balance these objectives (\autoref{fig:ablations}c).
