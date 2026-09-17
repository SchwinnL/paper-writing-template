# Example Method — Adversarial Training in LLMs

Paper: Continuous-Adversarial UL and Continuous-Adversarial IPO for adversarial training in LLMs.

---

\section{Method}
\label{method}
In this section, we introduce our adversarial training (AT) algorithms: Continuous-Adversarial UL (\advul) and Continuous-Adversarial IPO (\advdpo). We begin by reviewing the standard AT regime from~\citet{madry_towards_2018} (\S~\ref{sec:vis adv train}). We then explain differences between attacks in the standard AT setting and unique aspects of adversarial attacks in LLMs (\S~\ref{sec:threat model llm}). From there, we derive the Unlikelihood loss for---\advul{} (\S~\ref{sec:adv train llm}). Next, we introduce an adversarial \textsc{IPO} formulation---\advdpo{} (\S~\ref{sec:adv dpo}). Finally, we discuss key design decisions in the above AT algorithm (\S~\ref{sec:design decisions}).

\subsection{Adversarial Training}\label{sec:vis adv train}
AT is generally defined as a minimax optimisation problem as follows~\citep{madry_towards_2018}:
\begin{equation}\label{eq:at}
    \min_{\theta}\mathbb{E}_{(x,y)\in \mathcal{D}}\left[\max_{\delta\in T(x)} \mathcal{L}(f_{\theta}(x+\delta),y)\right],
\end{equation}
where $\mathcal{L}$ is the loss function, $f_{\theta}$ is a neural network with parameters $\theta$, $\mathcal{D}$ is the dataset, $T(x)$ is the set of perturbations around $x \in \mathcal{X}$ allowed by the threat model. In computer vision, $x \in [0,1]^d$ is an image, $T(x) = \{\delta  \mid \epsilon \geq \|\delta\|_p \,,\, x + \delta \in [0,1]^d\}$ and $\mathcal{L}$ is a classification loss such as cross-entropy.

\subsection{Attack Perturbation Sets in LLMs}\label{sec:threat model llm}
For LLMs with a token vocabulary $\mathcal{V}$, $x$ is a prompt and a common perturbation set $T$ are discrete manipulations of the input space, such as suffix attacks~\citep{zou2023universal}. For suffix attacks, the set of acceptable perturbations $\delta$ is defined to be in the set of sequences of tokens of length $m$ that can be appended to the input prompt. In other words, the adversarial attack $x + \delta$ is of the form $x; \delta$, where $\delta$ is a fixed number of tokens the attacker has full control over and $;$ means concatenation. However, computing the best $\delta$ from this perturbation set $T_{\mathrm{suffix}}(x) = \{\delta \mid x+\delta \in \mathcal{V}^{n+m}\}$ is computationally expensive, as the optimisation turns into a discrete combinatorial problem with exponentially many solutions. Arguably, it is too expensive to use during training, especially for large datasets.

Thus, we propose a different perturbation set $T$ based on continuous embedding attacks~\citep{schwinn2023adversarial}. This perturbation set allows the modification of the embeddings of the tokens in the prompt under some $\epsilon$-ball as measured under the $\ell_p$ norm. $E$ is a function from tokens $v\in \mathcal{V}$ to embeddings $E(v) \in \mathbb{R}^k$. We abuse notation and for a sequence $x = v_1;v_2;\ldots;v_n$ we say that $E(x) = E(v_1);E(v_2);\ldots;E(v_n)$. Our perturbation set allows for a $\delta_i\in\mathbb{R}^k$ around each token embedding. Therefore, the modified prompt after the attack $x + \delta$ is $E(v_1)+\delta_1;\ldots;E(v_n)+\delta_n$, where $\delta\in\mathbb{R}^{n\times k}$ and $T_{\mathrm{cont.}}(x) = \{\delta  \mid \forall i.\, \epsilon \geq \|\delta_i\|_p\,,x+\delta \in \mathbb{R}^{n\times k}\}$, as in the standard AT setting. \citet{schwinn2023adversarial} proposes to find the perturbation $\delta$ with signed gradient descent as in~\cite{goodfellow_explaining_2015}:
\begin{equation}\label{eq:adv iter}
    \delta^{t+1} = \delta^t + \alpha \cdot \mathrm{sign} (\nabla \log f(y|x+\delta^t)).
\end{equation}

\subsection{Adversarial Training in LLMs}\label{sec:adv train llm}

As described in Eq.~\ref{eq:at}, the inner loop of standard AT involves finding the worst-case perturbation by maximising the loss with respect to the ground truth prediction in an \emph{untargeted} way. In contrast, the goal of attacks on LLMs is to induce a specific harmful continuation $\hat{y}$ given a harmful prompt $x$. This exemplifies adversarial training under a \emph{targeted attack}.
\citet{mazeika2024harmbench} propose a loss that encourages the model to \emph{i)} increase the likelihood of a ``safe'' continuation $y$ (e.g.\ ``\texttt{I am sorry, ...}''), and \emph{ii)} decrease the likelihood of the unsafe continuation $\hat{y}$, given the targeted adversarial perturbation of $x$. This yields:
\begin{equation}\label{eq:ul}
    \min_{\theta} -\mathbb{E}_{(x,y,\hat{y})\in \mathcal{D}}\Bigl[\underbrace{\log f_{\theta}(y|x+\delta(x,\hat{y}))}_{\text{toward loss}}
- \underbrace{\log f_{\theta}(\hat{y}|x+\delta(x,\hat{y}))}_{\text{away loss}}\Bigr],
\end{equation}
where $\delta(x,\hat{y})=\argmin_{\delta'\in T(x)}\mathcal{L}(f(\hat{y}|x+\delta'))$ is the targeted attack on $x$. Contrary to standard AT~\citep{madry_towards_2018}, we are not maximising the loss of the safe answer, but specifically minimising towards a particular harmful continuation $\hat{y}$. As discussed in the previous section, $\delta$ naturally depends on the choice of $T,f,\mathcal{L}$, but we leave that out of the notation for clarity. Losses of the form of Equation~\ref{eq:ul} have been referred to as ``unlikelihood'' losses (\textsc{UL})~\citep{welleck2019neural,rafailov2024direct}. Note that the dataset $\mathcal{D}$ contains harmful prompts $x$ under which we want to give a safe answer $y$ rather than an unsafe answer $\hat{y}$.

In addition to the two terms in Equation~\ref{eq:ul}, \citet{mazeika2024harmbench} propose to add an additional loss term that maximises the utility of the model,~i.e.\ given an utility dataset $\mathcal{D}_{\mathrm{u}}$, it optimises:
\begin{equation}\label{eq:ul+utility}
\min_{\theta}
-\mathbb{E}_{(x,y,\hat{y})\in \mathcal{D}}\Bigl[\underbrace{\log f_{\theta}(y|x+\delta(x,\hat{y}))}_{\text{toward loss}}
- \underbrace{\log f_{\theta}(\hat{y}|x+\delta(x,\hat{y}))}_{\text{away loss}}\Bigr]
- \mathbb{E}_{(x,y)\in\mathcal{D}_{\mathrm{u}}}\Bigl[\underbrace{\log f_{\theta}(y|x)}_{\text{utility loss}}\Bigr],
\end{equation}

\citet{mazeika2024harmbench} found this loss necessary to avoid degenerate behaviours such as refusing to answer all prompts by producing some often generic refusal answer $y$.


\subsection{Continuous-Adversarial Unlikelihood}
The primary difference between \citet{mazeika2024harmbench} and our method is the choice of perturbation set used during AT. \citet{mazeika2024harmbench} choose \textbf{discrete} suffix attacks $T_{\mathrm{suffix}}$ and employ the \gcg{} algorithm along with several tricks to mitigate the computational cost to find a \gcg{} attack. One optimisation they introduce is to only update the attack after every $k$ training steps. In contrast, we employ $T_{\mathrm{cont.}}$ with \textbf{continuous} attacks as introduced by \citet{schwinn2023adversarial}, which are orders of magnitude ($ \times \efficency$) more efficient (see Table~\ref{tab:compute}). Consequently, we do not require any additional tricks to further reduce computational costs.
In the Unlikelihood loss (Eq~\ref{eq:ul}) we add cut-off values for the toward and away loss to prevent over-optimising either. Given a loss $\mathcal{L}'$ before, we implement the cutoff as $\mathcal{L} = \mathbb{I}[\mathcal{L}'>c] 0.999c + (\mathbb{I}[\mathcal{L}'>c]0.001 + \mathbb{I}[\mathcal{L}'\leq c])\mathcal{L'}$, where $c$ is the cutoff value chosen.

\subsection{Continuous-Adversarial IPO}\label{sec:adv dpo}
Equation~\ref{eq:ul} has a similar form to \dpo{}~\citep{rafailov2024direct}, which maximises the likelihood of a preferred answer while decreasing the likelihood of a dispreferred answer, given a prompt $x$. This motivates us to present the following loss function, which we will call Continuous-Adversarial \textsc{IPO} (\advdpo):
\begin{equation}\label{eq:adv dpo}
    \min_{\theta}-\mathbb{E}_{(x,y,\hat{y})\in \mathcal{D}}\left[\ell_{\beta}\left(
\log\frac{f_{\theta}(y|x+\delta(x,\hat{y}))}
    {f_{\theta_0}(y|x)}
- \log\,\frac{f_{\theta}(\hat{y}|x + \delta(x,\hat{y}))}
    {f_{\theta_0}(\hat{y}|x)}
\right)\right],
\end{equation}

where $\ell_{\beta}(h)$ would be the $\log\sigma(\beta h)$ in the original \dpo{}, but we use the loss proposed in \citet{azar2024general} called \textsc{IPO}, i.e.\ $\ell_{\beta}(h) = \left(h-\frac{1}{2\beta}\right)^2$, because it is less prone to overfitting.
This loss implicitly minimises the Kullback-Leibler divergence w.r.t.\ the original model distribution $f_{\theta_0}(y | x)$, which prevents the model to collapse to degenerate behaviors leading to refuse all prompts with the refusal answer $y$. As a result, we are able to omit the utility dataset for \advdpo.

\subsection{Design Decisions}\label{sec:design decisions}
A few design decisions worth discussing are:
\begin{enumerate}[leftmargin=0.5cm]
    \item The adversarial attack in the toward loss optimises $\delta$ such that the harmful output $\hat{y}$ becomes more likely. An alternative that we leave for future work would be to formulate the attack for the toward loss such that $y$ becomes less likely, i.e.\ $\delta(x,y)=\argmax_{\delta'\in T(x)}-\log (f(y|x+\delta'))$. It might even make sense to compute two separate attacks, one for $y$ and one for $\hat{y}$, and use them for the positive and negative cross-entropy loss terms, respectively. However, this would induce additional computational overhead.
    \item Importantly, we do not use the attack $\delta$ on the input for the reference model ($f_{\theta_0}$ in Equation~\ref{eq:adv dpo}). Empirically we found that this makes training unstable in the \dpo{} setting. We hypothesize that this is because the reference model represents roughly desirable log probability values of the safe answer $y$. Note that the original \dpo{} paper~\citep{rafailov2024direct} reports a similar observation and proposes to do \sft{} on the chosen continuation $y$ to make sure that these reference values are on-policy.
    \item \citet{mazeika2024harmbench} suggests to optimise $\log\, (1-f_{\theta}(\hat{y}|x+\delta(x,\hat{y})))$ instead of $-\log f_{\theta}(\hat{y}|x+\delta(x,\hat{y}))$ for the away loss. We explored this and found that it yielded a considerably worse robustness/safety trade-off. We were unable to find a model that is robust and maintains some level of utility.
\end{enumerate}
