### LaTeX Best Practices

**Citations:**
- Only ever cite sth with a concrete reference yourself if this reference is in the bib file. Very important
- Use `\citet{}` when authors are part of the sentence: "Smith (2001) shows..."
- Use `~\citep{}` or `~\cite{}` otherwise (depending on what is used in the document): "...recent work~\citep{smith2001}"
- Never write "(Smith, 2001) shows..."
- Acronym + citation: "proximal policy optimisation~\citep[PPO]{schulman2017ppo}"
- Cite the correct version—Google Scholar often defaults to arXiv rather than the conference paper

**Equations:**
- Display equations can take up space if overused; too many inline equations hurt readability
- Think carefully about which equations are worth displaying
- If you leave a blank line after `\end{equation}` or `$`, you create an extra line break
- Use `\stackrel{}` on non-trivial equality/inequality statements and justify immediately after
- Mathematical equations follow standard punctuation rules—don't forget periods and commas after equations

**Cross-references and packages:**
- Use `\usepackage[backref=page]{hyperref}` to make it easier to jump to references and back
- Use the cleveref package for intelligent cross-referencing (`\cref{}`)
- Use `\usepackage[acronym]{glossaries}` or `\usepackage{acronym}` to auto-manage acronyms
- DON'T use the fullpage package—it overrides options in many style files

**Formatting:**
- Use correct quotation marks: `` ``correct quotation'' `` or `\enquote{correct quotation}` from csquotes. Never use "" to quote something.
- Footnotes should be after "." and ","
- Make sure `\label{}` comes after `\caption{}` in figures
- Use `\left(` and `\right)` initially, but consider explicit sizes (`\big(`, `\Big(`, `\bigg(`) in final pass
- Check for broken references (indicated by ??)
- Minimize visual white space; fill the page limit
- Avoid line breaks resulting in lines with just a single word