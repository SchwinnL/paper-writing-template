# Example Proofs — Diffusion Adversarial Attacks

Paper: Using diffusion LLMs for adversarial prompt generation. Probabilistic bound for conditional sampling.

---

\section{Probabilistic Bound for Conditional Sampling}\label{app:bound}

\subsection{Setup}
Let $\vy^\star$ be a target response and $\mathcal{X}$ be the set of admissible prompts. We define the set of prompts that achieve an expected reward of at least $t$ under the true data distribution $q(\cdot|\vx)$ as:
\begin{equation}
    S_t = \{\vx \in \mathcal{X} : r_q(\vx) \ge t \},
\end{equation}
where $r_q(\vx) = \E_{\vy \sim q(\cdot \mid \vx)}[\mathrm{Reward}(\vy, \vx)].$
Similarly, the set of high-reward prompts under the target model $P_{f_{\theta_t}}$ is:
\begin{equation}
    S_t^{(\mathrm{t})} = \{\vx \in \mathcal{X} : r_{\mathrm{t}}(\vx) \ge t \},
\end{equation}
where $r_{\mathrm{t}}(\vx) = \E_{\vy \sim P_{f_{t}}(\cdot \mid \vx)}[\mathrm{Reward}(\vy, \vx)]$.
Our goal is to lower-bound the success probability $\Pr(\max_{i\le N} r_{\mathrm{t}}(\tilde{\vx}_i) \ge t)$ for $N$ i.i.d. samples $\tilde{\vx}_i \sim p_\theta(\vx \mid \vy^\star)$.

\paragraph{Assumptions.}
\begin{enumerate}
    \item \textbf{Surrogate Fidelity:} \\$TV\big(q(\vx \mid \vy^\star), p_\theta(\vx \mid \vy^\star)\big) \le \varepsilon_1$.
    \item \textbf{Target Fidelity:} \\$TV\big(q(\vy \mid \vx), P_{f_{t}}(\vy \mid \vx)\big) \le \varepsilon_2$ for all $\vx$.
    \item \textbf{Bounded Reward:} $\mathrm{Reward}(\vy, \vx) \in [0, 1]$.
\end{enumerate}

\subsection{Probabilistic Bound}
\begin{lemma}[Bounding the Expected Reward Difference]
\label{lem:reward_bound}
Under the target fidelity and bounded reward assumptions, the difference in expected rewards is bounded by $\varepsilon_2$:
\[ |r_{\mathrm{t}}(\vx) - r_q(\vx)| \le \varepsilon_2, \quad \forall \vx \in \mathcal{X}. \]
\end{lemma}
\begin{proof}
A standard property of total variation distance \citep{gibbs2002choosingboundingprobabilitymetrics} states that for any function $g$ with range $[a, b]$, it holds that $|\E_p[g] - \E_q[g]| \le (b-a) TV(p, q)$. Thus,
\[ |r_{\mathrm{t}}(\vx) - r_q(\vx)| \le (1-0) \cdot TV(P_{f_{t}}(\cdot \mid \vx), q(\cdot \mid \vx)) \le \varepsilon_2. \qedhere \]
\end{proof}

\begin{lemma}[Set Inclusion]
\label{lem:set_inclusion}
The set of high-reward prompts under the target model $S_t^{(\mathrm{t})}$ contains the set of slightly-higher-reward prompts under the true distribution:
\[ S_{t+\varepsilon_2} \subseteq S_t^{(\mathrm{t})}. \]
\end{lemma}
\begin{proof}
Let $\vx \in S_{t+\varepsilon_2}$. By definition, $r_q(\vx) \ge t + \varepsilon_2$. From Lemma \ref{lem:reward_bound}, we know that $r_{\mathrm{t}}(\vx) \ge r_q(\vx) - \varepsilon_2$. Combining these inequalities, we get:
\[ r_{\mathrm{t}}(\vx) \ge (t + \varepsilon_2) - \varepsilon_2 = t. \]
Therefore, by definition, $\vx \in S_t^{(\mathrm{t})}$.
\end{proof}

\begin{theorem}[Probabilistic Lower Bound on Success]
Let $\tilde{\vx}_1, \dots, \tilde{\vx}_N$ be i.i.d. samples from the surrogate $p_\theta(\cdot \mid \vy^\star)$. The probability of finding at least one prompt with target reward $\ge t$ is lower-bounded by:
\[ \Pr\Big(\max_{i\le N} r_{\mathrm{t}}(\tilde{\vx}_i) \ge t\Big) \ge 1 - \left(1 - \left(q(S_{t+\varepsilon_2} \mid \vy^\star) - \varepsilon_1\right)\right)^N, \]
provided that $q(S_{t+\varepsilon_2} \mid \vy^\star) \ge \varepsilon_1$.
\end{theorem}
\begin{proof}
The probability of success is the complement of all $N$ samples failing:
\begin{equation}
\begin{split}
    \Pr\Big(\max_{i\le N} r_{\mathrm{t}}(\tilde{\vx}_i) \ge t\Big) = 1 - \Pr\left(\forall i, \tilde{\vx}_i \notin S_t^{(\mathrm{t})}\right) \\= 1 - \big(1 - p_\theta(S_t^{(\mathrm{t})} \mid \vy^\star)\big)^N.
\end{split}
     \label{eq:success_prob}
\end{equation}
To find a lower bound on this probability, we need a lower bound for $p_\theta(S_t^{(\mathrm{t})} \mid \vy^\star)$. Using the set inclusion from Lemma \ref{lem:set_inclusion}:
\[ p_\theta(S_t^{(\mathrm{t})} \mid \vy^\star) \ge p_\theta(S_{t+\varepsilon_2} \mid \vy^\star). \]
Next, we apply the surrogate fidelity assumption. For any event $A$, $|p_\theta(A \mid \vy^\star) - q(A \mid \vy^\star)| \le \varepsilon_1$, which implies $p_\theta(A \mid \vy^\star) \ge q(A \mid \vy^\star) - \varepsilon_1$. Applying this to the set $S_{t+\varepsilon_2}$:
\[ p_\theta(S_{t+\varepsilon_2} \mid \vy^\star) \ge q(S_{t+\varepsilon_2} \mid \vy^\star) - \varepsilon_1. \]
Combining these inequalities gives us the required lower bound on the single-trial success probability:
\[ p_\theta(S_t^{(\mathrm{t})} \mid \vy^\star) \ge q(S_{t+\varepsilon_2} \mid \vy^\star) - \varepsilon_1. \]
Substituting this back into Equation \eqref{eq:success_prob} yields the final result.
\end{proof}
