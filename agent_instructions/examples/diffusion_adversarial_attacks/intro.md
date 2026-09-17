# Example Introduction — Diffusion Adversarial Attacks

Paper: Using diffusion LLMs for adversarial prompt generation.

---

\section{Introduction}

\glspl{llm} achieve impressive generalization across a wide range of language tasks, yet remain highly sensitive to perturbations in their input prompts.
This sensitivity enables \emph{adversarial attacks}, i.e., constructing inputs that induce a targeted, often undesired response.
Despite a rapidly growing literature on attacks~\cite{zou2023universal, geisler2024attacking}, current methods are often inefficient and unreliable, fall short of manual human red-teaming~\cite{li2024llm, nasr2025attacker}, and underperform compared to attacks in other domains such as computer vision~\cite{szegedy_intriguing_2014}.

We argue that a major reason for the inefficiency of current attacks is the \textit{autoregressive nature} of most deployed \glspl{llm}.
Autoregressive models parameterize $q(\vy\mid\vx)$, the distribution of responses conditioned on prompts, whereas adversarial prompting requires a solution to the inverse problem: finding prompts $\vx$ that produce a desired response $\vy$.
Because autoregressive models do not allow direct inference of $q(\vx\mid\vy)$, many existing attacks rely on indirect search or heuristic optimization in discrete token space, which can be computationally costly and unreliable~\cite{li2024llm, schwinn2025adversarial, beyer2025llm}.


To overcome the inefficiency of existing attacks, we leverage models that learn the \emph{joint distribution} $q(\vx, \vy)$ over prompt–response pairs.
This allows inference of the conditional $q(\vx\mid\vy)$ and enables direct generation of prompts likely to elicit a desired response.
\glspl{dllm} naturally realize this idea by modeling $(\vx,\vy)$ jointly rather than autoregressively, allowing for \textit{inpainting}-like conditioning.
By fixing the target response $\vy^\star$ throughout the standard generative diffusion process, one can invert the conditional and effectively sample candidate adversarial prompts from $p_\theta(\vx\mid\vy^\star)$ (cf.\ \cref{fig:figure1}).

\begin{figure}[t!]
    \centering
  \includegraphics[width=\linewidth, trim={5.2cm 1.5cm 8.cm 4.5cm},clip]{plots/figure1.pdf}\caption{We present \method, a novel framework that reformulates the costly and iterative process of finding adversarial prompts into a simple inference task leveraging \emph{pretrained} \glspl{dllm}.}\label{fig:figure1}
\end{figure}

We formally show that, under mild fidelity assumptions on the surrogate and target model, only a small number of conditional samples are required to recover high-reward prompts.
This theoretical insight provides a probabilistic guarantee and establishes diffusion-based amortized inference as a principled and model-agnostic framework for adversarial prompt generation.
By sampling conditionally, we can efficiently obtain attacks that transfer across multiple black-box target models, transforming a previously costly search problem into a parallelizable inference task.

Our main contributions can be summarized as follows:
\begin{itemize}[leftmargin=10pt,rightmargin=10pt,labelsep=4pt,topsep=4pt,itemsep=4pt,parsep=0pt]
    \item \textbf{Amortized prompt search:} We propose \method, a novel framework that transforms costly per-instance optimization into conditional inference using pretrained, non-autoregressive \glspl{llm} (e.g., \glspl{dllm}) as surrogates.
    \item \textbf{Prompt discovery guarantees:} We derive probabilistic guarantees that a small number of samples suffices to recover high-reward prompts under mild fidelity assumptions on the target and surrogate model.
    \item \textbf{Efficient and transferable attacks:} Experimentally we show that our method generates low perplexity, adversarial prompts that succeed across black-box \glspl{llm}, including robustly trained and proprietary models, at a fraction of existing attacks cost.
\end{itemize}
