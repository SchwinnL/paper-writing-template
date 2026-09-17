# Example Introduction — Position Paper on Adversarial Alignment

Paper: "Position: We Need to Realign Incentives for Meaningful Progress in Adversarial Alignment for LLMs" (ICML 2026).

---

\section{Introduction}

Security risks in computer science have been a prevalent issue for decades~\cite{valiant_learning_1985, kearns_learning_1993}.
This ongoing challenge has resulted in an ``arms race" that includes the continued development of new attacks, such as malware and phishing, as well as defense mechanisms.

In the field of deep learning, \citet{szegedy_intriguing_2014} discovered that deep neural networks are highly susceptible to \textit{adversarial examples} -- input perturbations optimized to mislead models into making predictions that are erroneous or misaligned with their intended behavior. In response, countless defense strategies were proposed to safeguard neural networks against these attacks in the last decade. Yet, most newly proposed heuristic defenses were exposed as flawed by subsequent evaluations, often by using standard attack protocols that already existed at the time of the defense publication~\citep{tramer_adaptive_2020}. 

We investigate the research state of adversarial alignment in large language models (LLMs), which we define as the ability of an aligned model to maintain its intended training objective in the presence of adversarial attacks. 
We observe that adversarial alignment risks following the same cycle of flawed defenses and subsequent rectified evaluations seen in past adversarial robustness research but with considerably amplified challenges and stakes~\cite{hendrycks2022x, zou2023universal, schwinn2023adversarial}. 
Unlike previous robustness research that generally focused on well-defined problems like image classification~\cite{goodfellow_explaining_2015}, assessing LLM capabilities is considerably more challenging due to inherent ambiguities of the alignment problem and the complexity of natural language~\cite{wolf2023fundamental, andriushchenko2024jailbreaking, li2024llm}. Furthermore, the potential harm associated with LLMs is substantially greater due to their advanced capabilities and widespread availability~\cite{hendrycks2022x} in particular as they start being used as autonomous agents.

\begin{table*}[t!]
    \centering
    \caption{Non-exhaustive comparison of robustness research in previous domains and LLMs.}
     \newcommand{\cellwidth}{2.2cm}
     \newcommand{\linesp}{3pt}
     \newcommand{\cbox}[1]{\parbox[t][1cm][t]{\cellwidth}{\centering #1}}
    \tiny	
\begin{tabular}{l*{3}{>{\centering\arraybackslash}m{\cellwidth}}*{3}{>{\centering\arraybackslash}m{\cellwidth}}}
    \addlinespace
     & \multicolumn{3}{@{\hspace{3mm}}c@{\hspace{3mm}}}{\cellcolor{customblue}\textcolor{white}{\textbf{Attack goals}}} & \multicolumn{3}{@{\hspace{3mm}}c@{\hspace{3mm}}}{\cellcolor{customblue}\textcolor{white}{\textbf{Attack Capabilities}}} \\
    \addlinespace[\linesp]
    \rowcolor{darkgray} & \textbf{Objectives} & \textbf{Attacks} & \textbf{Datasets} & \textbf{Access} & \textbf{Constraints} & \textbf{Frameworks} \\
    \rowcolor{lightgray}\textbf{Previous} & \cbox{Clear objectives\\(e.g., classification)\\\cite{szegedy_intriguing_2014, madry_towards_2018}} & \cbox{Generally reliable\\\cite{carlini_evaluating_2019, tramer_adaptive_2020, croce2020reliable}} & \cbox{Standardized\\\cite{croce2020reliable, Schwinn2021Jitter, croce2020robustbench}} & \cbox{Generally white-box\\and open-source\\\cite{szegedy_intriguing_2014, croce2020robustbench}} & \cbox{Tractable but incomplete\\(e.g., $\ell_p$)~\cite{szegedy_intriguing_2014, goodfellow_explaining_2015, madry_towards_2018}} & \cbox{Standardized\\\cite{croce2020robustbench, papernot2018cleverhans, rauber2017foolbox}} \\
    \addlinespace[\linesp]
    \rowcolor{lightgray}\textbf{LLM} & \cbox{Alignment \& robustness entangled~\cite{zou2023universal, mazeika2024harmbench}} & \cbox{Currently weak\\\cite{andriushchenko2024jailbreaking, li2024llm}} & \cbox{Entangled notions of harmfulness\\\cite{mazeika2024harmbench, chao2024jailbreakbench}} & \cbox{Often black-box and\\proprietary models\\\cite{zou2023universal, chao2023jailbreaking}} & \cbox{No constraints\\\cite{zou2023universal, mazeika2024harmbench, chao2023jailbreaking, zhu2023autodan}} & \cbox{Varying evaluation settings\\\cite{mazeika2024harmbench, chao2024jailbreakbench}} \\
\end{tabular}
    \label{tab:comparison}
    \vspace{-15pt}
\end{table*}

We argue that the past lack of progress can be largely attributed to a narrow focus on improving benchmark numbers without sufficient attention to rigorous evaluations and clear evaluation criteria. This led to the proliferation of ad-hoc defenses that relied on security through obscurity and ultimately proved ineffective, thus not providing a solid foundation for future work. Now, the field of adversarial alignment risks repeating the same mistakes, expending significant effort but failing to make meaningful progress.  
Our position on adversarial alignment in LLMs is as follows:

\begin{tcolorbox}[
    colback=white, colframe=customblue, coltitle=white, fonttitle=\bfseries, 
    rounded corners, enhanced, 
    title=Position, 
    attach boxed title to top left={yshift=-2mm, xshift=5mm}, 
    boxed title style={colback=customblue, rounded corners},
    boxsep=1mm,
    left=1mm,
    right=1mm
]
 We argue that researchers deal with:\,\,\textbf{1)} vague problem definitions, where alignment and robustness are inherently entangled in the adversarial alignment problem, \textbf{2)} complex, non-reproducible evaluations, and \textbf{3)} emphasis on state-of-the-art attack performance against proprietary models over reproducible and comparable open-source research. \textbf{Thus, meaningful progress in adversarial alignment for LLMs requires simpler, reproducible, and more measurable objectives.}
\end{tcolorbox}

Our main contributions are:

\begin{itemize}[noitemsep,nolistsep,topsep=-2pt,leftmargin=0.6cm] 
    \item We systematically identify challenges in previous robustness research, how they apply to LLMs, and how new challenges emerged. Based on this analysis: 
    \item We demonstrate how adversarial alignment intertwines the challenges of alignment and robustness, making robustness evaluation difficult, as it inherits the challenges of measuring alignment, such as ambiguity in success criteria. 
    We advocate for a complementary approach that directs the majority of technical works toward simpler sub-problems with measurable objectives, while reserving different criteria for exploratory research.
    
    \item Towards the same goal of improving measurability, we propose simplifying robustness benchmarks by evaluating specific types of harm individually rather than combining complex concerns like copyright infringement, fairness, and toxicity into a single evaluation.
    
    \item We emphasize the need for academia to prioritize reproducible research over chasing SOTA performance on proprietary models. In this context, we propose fostering open-source research with accessible models. 
    
    \item Computational overhead, vast number of hyperparameters, and varying implementation details hinder comparability between different works. We advocate for a practical approach to improve reproducibility and comparability: community-driven leaderboards and standardized benchmarks to encourage best practices in adversarial robustness research.  
\end{itemize}

For a concise discussion and supporting literature regarding the claims made in this work, we refer the reader to Appendix~\ref{app:evidence}.

\section{Structure of our argument}\label{sec:structure}

We structure our position as follows: We define a taxonomy of adversarial robustness threats based on common cybersecurity frameworks including \textbf{I)} robustness goals and \textbf{II)} adversary capabilities. For both elements of this taxonomy, we follow a parallel argument structure: We \textbf{A)} define past and present threat models (see start of \S\ref{sec:pos-goals} and \S\ref{sec:pos-capability}), \textbf{B)} discuss historical challenges and connections to upcoming and exacerbated issues in the LLM domain (\S\ref{sec:pos-goals-challenges}, and \S\ref{sec:pos-capability-challenges}), and \textbf{C)} explore the applicability of past insights to new problems and how current research objectives can be realigned to promote measurable progress (\S\ref{sec:pos-goals-realigned} and \S\ref{sec:pos-capability-realigned}). We provide a non-exhaustive comparison between past and current robustness research in Table~\ref{tab:comparison}. (Appendix~\ref{app:evidence}). \\
Additionally, Appendix~\ref{app:other_works} presents a comprehensive comparison between our work and other contributions from the AI safety community, including differences to the concurrently submitted position paper by~\citet{rando2025adversarial}.
