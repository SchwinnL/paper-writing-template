# Example Method — Diffusion Adversarial Attacks

Paper: Using diffusion LLMs for adversarial prompt generation.

---

\section{Method}
Our method reframes the expensive optimization problem of finding adversarial prompts as an efficient inference task.
We show that a non-autoregressive, pretrained generative \gls{llm}, such as a \gls{dllm}, can serve as a powerful generative prior over plausible prompt–response pairs, enabling conditional generation of adversarial prompts for a given target response.
By sampling conditionally on a target response, we obtain diverse, high-quality candidate prompts that tend to elicit desired responses from a variety of black-box target \gls{llm}.
This amortized formulation replaces costly per-instance optimization with a small number of parallelizable samples from a pretrained model.

\textbf{Optimization Objective.}
Given a target (potentially black-box) \gls{llm} $f$ that maps prompts to a distribution over responses $\target(\vy \mid \vx)$, our goal is to find a prompt that maximizes a scalar reward function.
The reward function, $\mathrm{Reward}(\vy,\tilde{\vx}) \in [0,1]$, measures the success of a generated response $\vy$ elicited by a prompt $\tilde{\vx}$.
We define the expected reward under both the target model and the true data distribution as:
\begin{align}\label{eq:expected_reward}
    r_{\mathrm{t}}(\vx) &= \E_{\vy\sim \target(\cdot\mid \vx)}[\mathrm{Reward}(\vy,\vx)],
    \\
    r_q(\vx) &= \E_{\vy\sim q(\cdot\mid \vx)}[\mathrm{Reward}(\vy,\vx)].
\end{align}

Then the optimization objective is:
\begin{equation}\label{eq:target}
\tilde{\vx}^\star = \arg\max_{\tilde{\vx} \in \Phi(\vx)}
\E_{\vy \sim \target(\cdot \mid \tilde{\vx})}
\big[ \mathrm{Reward}(\vy, \tilde{\vx}) \big],
\end{equation}
where $\Phi(\vx) \subseteq \mathcal{X}$ is the set of admissible prompts (typically $\Phi(\vx) = \mathcal{X}$, with soft constraints imposed through the reward function).

Note that the expected reward \smash{$r_{\mathrm{t}}(\vx)$} (\cref{eq:expected_reward}) can be interpreted as an \emph{unnormalized probability mass function} over prompts, assigning higher mass to inputs that elicit desirable responses from the target model.
This defines a reward-weighted posterior $\pi(\vx) \propto r_{\mathrm{t}}(\vx)$, from which sampling corresponds to drawing adversarially successful prompts, with $\tilde{\vx}^\star$ corresponding to the mode.
Thus, intuitively our objective is to generate prompts $\vx$ that elicit high-reward responses from the target model $f$.

\subsection{Amortized Search via Surrogate Model}\label{sec:armotized_search}

Directly optimizing \cref{eq:target} over the discrete space of prompts $\mathcal{X}$ is computationally prohibitive.
Our key insight is to solve this optimization via amortized search \citep{amos2023tutorial}, using a surrogate generative model $\model(\vx \mid \vy) \approx q(\vx \mid \vy)$ to approximate $\pi(\vx)$.
This surrogate provides a generative shortcut: sampling from $\model(\vx\mid\vy)$ for a fixed response $\vy$ yields candidate prompts that are likely under the true data distribution, replacing iterative optimization with efficient conditional inference.
Thus, instead of searching for a good prompt, we can simply sample one.

To formally connect $\model(\vx\mid\vy)$ to the optimization objective, we make the following assumptions about the fidelity of the target and surrogate model around a target response $\vy^\star$:
\begin{enumerate}
    \item \textbf{Surrogate Fidelity:} The surrogate conditional distribution is close to the true data conditional:\\
    $\mathrm{TV}(q(\vx \mid \vy^\star), p_\theta(\vx\mid \vy^\star)) \le \varepsilon_1$.
    \item \textbf{Target Fidelity:} The target model response distribution is close to the true data conditional for all prompts:\\
    $\mathrm{TV}(q(\vy^\star\mid \vx), \target(\vy^\star\mid \vx)) \le \varepsilon_2$ for all $\vx$.
\end{enumerate}
Here, $\mathrm{TV}(\cdot, \cdot)$ is the total variation distance.
These assumptions are reasonable for models trained to minimize \gls{kl} divergence (i.e., via maximum likelihood), as Pinsker's inequality bounds the symmetric TV by the \gls{kl} divergence.

\begin{figure}[H]
    \centering
  \includegraphics[width=0.8\linewidth, trim={0.5cm 3.5cm 18.cm 3.6cm},clip]{plots/diffusion_llm.pdf}\caption{Where the surrogate $\model(\vx \mid \vy^{\star})$ meets high expected reward under a black-box target model $\target(\vy \mid \vx)$.}\label{fig:figure2}
\end{figure}
\textbf{Success Probability.}
Lets have a look at the set of prompts that achieve an expected reward of at least $t$ under the data distribution for a fixed target response $\vy^\star$: $S_t = \{\vx : r_q(\vx) \ge t\}$.
The conditional probability mass of this set under the true data distribution is $\alpha = q(S_t\mid \vy^\star)$.

Then the probability that at least one of $N$ i.i.d. samples $\tilde{\vx}_i \sim p_\theta(\cdot \mid \vy^\star)$ achieves the desired reward threshold:
\begin{equation}
\begin{split}
  \Pr\!\Big(\max_{i\le N} r_{\mathrm{t}}(\tilde{\vx}_i)\ge t\Big)
  \approx \Pr\!\Big(\max_{i\le N} r_{\mathrm{q}}(\tilde{\vx}_i)\ge t\Big) \\
  = 1 - \Pr\left(\forall i,\, \tilde{\vx}_i \notin S_t\right)
  \approx 1 - (1-\alpha)^N.
\end{split}
\end{equation}
where $\alpha = q(S_t\mid \vy^\star) \approx p_\theta(S_t\mid \vy^\star)$, if the surrogate and target models are well-calibrated ($\varepsilon_1 \approx 0, \varepsilon_2 \approx 0$). In \cref{app:bound} we provide a probabilistic bound for $\varepsilon_1, \varepsilon_2 >0$.

This result formalizes our core intuition: if under the data distribution the target response $\vy^\star$ co-occurs with high-reward prompts with a non-negligible fraction $\alpha$, then only a modest number of samples from the surrogate $p_\theta(\vx \mid \vy^\star)$ is needed to find a prompt with high expected adversarial reward. The surrogate can thus act as an \emph{amortized optimizer}, replacing costly search with efficient sampling.


\subsection{Conditional Prompt Generation}
Many non-autoregressive \glspl{llm} (e.g., flow \citep{havasi2025editflowsflowmatching} and diffusion \citep{zhu2025llada15variancereducedpreference,nie2025largelanguagediffusionmodels,ye2025dream7bdiffusionlarge}) trained on text sequences $(\vx, \vy)$ implicitly learn the joint distribution $\model(\vx, \vy) \approx q(\vx, \vy)$.
For those the conditional surrogate $p_\theta(\vx \mid \vy)$ required for our method can be derived directly from the learned joint via Bayes' rule:
\begin{equation}
    p_\theta(\vx \mid \vy^\star)
= \frac{p_\theta(\vx, \vy^\star)}{p_\theta(\vy^\star)}
\approx \frac{q(\vx, \vy^\star)}{q(\vy^\star)}
= q(\vx \mid \vy^\star).
\end{equation}
Moreover, for a fixed \(\vy^\star\), maximizing \(q(\vx \mid \vy^\star)\) (or equivalently, \(q(\vx, \vy^\star)\)) favors prompts that are most likely to co-occur with that response in the data.
If high joint likelihood correlates with high reward under the target model, these conditionally sampled prompts are natural candidates for maximizing the adversarial objective.

In this paper we focus on common \glspl{dllm}, specifically \cite{nie2025largelanguagediffusionmodels}, where sampling from $p_\theta(\vx \mid \vy^\star)$ can be achieved via inpainting-like conditional sampling, similar to the conditioning proposed for, e.g., images \citep{lugmayr2022repaintinpaintingusingdenoising, rout2023theoreticaljustificationimageinpainting, lienen2024zeroturbulencegenerativemodeling}, language \citep{gat2024discreteflowmatching}, graphs \citep{ketata2025jointrelationaldatabasegeneration}, sequences \citep{editpp} and sets \citep{psdiff}.
The generation process starts with random noise for the entire sequence $\vz_T = (\vx_T, \vy_T) \sim p(\vz_T)$, which is iteratively denoised using the learned reverse Markov kernels $p_\theta(\vz_{t-1} \mid \vz_t)$.
Then conditional sampling boils down to simulating the reverse diffusion chain
\begin{equation}
    \vz_{t-1}\!\sim\!p_\theta(\vz_{t-1}\mid\vz_t),\qquad \vz_t=(\vx_t,\vy_t),
\end{equation}
while \emph{overwriting} the response with $\vy_{t-1}\!\leftarrow\!\vy^\star$ at each step $t=T,\dots,1$.
This procedure projects the joint diffusion trajectory onto the manifold where the response is fixed, yielding an approximate sample $\tilde\vx=\vx_0 \sim p_\theta(\vx \mid \vy^\star)$.

\subsection{Guided Conditional Sampling}\label{sec:guided-conditional-sampling}
To further improve sampling efficiency, we can additionally bias the generation process towards high-reward prompts using guidance \citep{dhariwal2021diffusionmodelsbeatgans}.
At each denoising step $t$, we bias the sampling distribution by reweighting each candidate according to a scoring function:
\begin{equation}
    \tilde{p}_\theta(\vz_{t-1}\mid \vz_t)
\;\propto\;
p_\theta(\vz_{t-1}\mid \vz_t) \,
\mathrm{Score}(\vx_{t-1}, \vy^\star),
\end{equation}
where the score acts as an importance weight.
We consider two complementary scoring functions:
\begin{enumerate}
    \item \textbf{Likelihood Guidance:} $\mathrm{Score}(\vx, \vy^\star) = \target(\vy^\star \mid \vx)$. This steers generation towards prompts that the target model already considers highly likely to produce the response $\vy^\star$.

    \item \textbf{Reward Guidance:} $\mathrm{Score}(\vx, \vy^\star) = \mathrm{Reward}(\vy^\star, \vx)$. This directly optimizes for adversarial success by guiding the generation towards prompts that yield a high reward.
\end{enumerate}

While the reward guidance is natural, the Likelihood guidance is evident if one considers the idealized reward for eliciting a specific response \(\vy^\star\),
\(\mathrm{Reward}(\vy,\vx) = \mathbb{I}(\vy = \vy^\star)\).
Then the expected reward equals the target likelihood:
\begin{equation}
    \E_{\vy\sim \target(\cdot\mid\vx)}[\mathrm{Reward}(\vy,\vx)] = \target(\vy^\star \mid \vx).
\end{equation}

In practice, we can sample from a guided model, by sampling $k$ times from the diffusion model $p_\theta(\vz_{t-1}\mid \vz_t)$ and retaining the sample with the highest score.



\subsection{Summary}

Our method demonstrates that pretrained \glspl{llm} modeling the joint distribution $p_\theta(\vx,\vy)$ can serve as a generative prior over realistic prompt–response pairs, transforming adversarial prompt search into an amortized inference problem.
Under mild fidelity assumptions—that both the diffusion and target models approximate the true data distribution—conditional sampling from $p_\theta(\vx\mid\vy^\star)$ yields prompts concentrated in regions of high joint likelihood with the target response $\vy^\star$.
Moreover, if $\vy^\star$ co-occurs with a non-negligible fraction of high-reward prompts in the data distribution, only a modest number of conditional samples suffices to recover these high-reward candidates.

Notably, because the surrogate models the underlying data manifold rather than optimizing for any specific target model, the same conditional samples $\{\tilde{\vx}_i\}$ can be amortized across multiple target models $f$.
This enables reusable, model-agnostic adversarial prompt generation, and reduces the cost of prompt optimization across target \glspl{llm}.
