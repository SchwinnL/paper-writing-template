# Example Background — Diffusion Adversarial Attacks

Paper: Using diffusion LLMs for adversarial prompt generation.

---

\subsection{Problem Set-up and Notation}
Let $\mathcal{T}$ be a discrete token vocabulary and $\mathcal{X}=\bigcup_{n=0}^{N}\mathcal{T}^n$ the set of all token sequences up to length $N$.
We write a sequence as the concatenation of a \emph{prompt} $\vx=(x_1,\dots,x_{n_X})$ and a \emph{response} $\vy=(y_1,\dots,y_{n_Y})$, and treat $(\vx,\vy)\in\mathcal{X}$ as a single joint sequence when convenient.

Assume a \gls{llm} modelling the joint distribution $\model(\vz)$ over sequences $\vz\in\mathcal{X}$, which is pretrained to approximate the true data-generating distribution $q(\vz)$.
When necessary, we will distinguish the prompt and response components, writing $\model(\vx,\vy)$ for the joint distribution and $\model(\vx \mid \vy)$ for the conditional distribution.
