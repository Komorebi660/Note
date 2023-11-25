# Latex Grammer

- [Latex Grammer](#latex-grammer)
  - [Mathematical Symble](#mathematical-symble)
  - [Title \& Packages](#title--packages)
  - [Sections](#sections)
  - [Enumerate](#enumerate)
  - [Table](#table)
  - [Figure](#figure)
  - [Algorithm](#algorithm)
  - [Equation](#equation)

## Mathematical Symble

| code | result |
| :--: | :----: |
| `\tilde` | $\tilde{a}$ |
| `\overline` | $\overline{ab}$ |
| `\underline` | $\underline{ab}$ |
| `\underbrace_{1}` | $\underbrace{a+b+c}_{1}$ |
| `bar` | $\bar{ab}$ |
| `\hat` | $\hat{ab}$ |
| `\vec` | $\vec{ab}$ |
| `\overrightarrow` | $\overrightarrow{ab}$ |
| `\dot` | $\dot{a}$ |
| `\ddot` | $\ddot{a}$ |
| `\sum_{}^{}{}` | $\sum_{}^{}{}$ |
| `\prod_{}^{}{}` | $\prod_{}^{}{}$ |
| `\int_{}^{}{}` | $\int_{}^{}{}$ |
| `\sqrt[3]{}` | $\sqrt[3]{a}$ |
| `\infty` | $\infty$ |
| `\sim` | $\sim$ |
| `\approx` | $\approx$ |
| `\propto` | $\propto$ |
| `\equiv` | $\equiv$ |
| `\neq` | $\neq$ |
| `\subset` | $\subset$ |
| `\supset` | $\supset$ |
| `\cup` | $\cup$ |
| `\cap` | $\cap$ |
| `\in` | $\in$ |
| `\notin` | $\notin$ |
| `\forall` | $\forall$ |
| `\exists` | $\exists$ |
| `\emptyset` | $\emptyset$ |
| `\lfloor` | $\lfloor$ |
| `\rfloor` | $\rfloor$ |
| `\lceil` | $\lceil$ |
| `\rceil` | $\rceil$ |
| `partial` | $\partial$ |
| `\nabla` | $\nabla$ |
| `dag` | $\dag$ |
| `ddag` | $\ddag$ |
| `\gg` | $\gg$ |
| `\ll` | $\ll$ |

## Title & Packages

```latex
\documentclass[12pt]{article}
\usepackage[UTF8, scheme = plain]{ctex}
\usepackage{indentfirst}
\usepackage{multirow}
\usepackage{threeparttable}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{tikz}
\usepackage{verbatim}
\usepackage{amsmath, bm}
\usepackage{mathrsfs}
\usepackage{amssymb}
\usepackage{listings}
\lstset{basicstyle=\ttfamily}
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}
\usepackage{wrapfig}
\usepackage[colorlinks,
            linkcolor=blue,
            anchorcolor=blue,
            citecolor=blue]{hyperref}
\usepackage{caption}
\usepackage{libertine}
\captionsetup[figure]{font=small, labelfont={bf,it}, name={图}}
\captionsetup[table]{font=small, labelfont={bf,it}, name={表}}

\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\lstset{frame=tb,
  language=C++,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=flexible,
  basicstyle={\small\ttfamily},
  numbers=none,
  numberstyle=\tiny\color{dkgreen},
  keywordstyle=\color{blue},
  commentstyle=\color{gray},
  stringstyle=\color{mauve},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=3
}

\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}

\makeatletter
\newenvironment{breakablealgorithm}
  {% \begin{breakablealgorithm}
   \begin{center}
     \refstepcounter{algorithm}% New algorithm
     \hrule height.8pt depth0pt \kern2pt% \@fs@pre for \@fs@ruled
     \renewcommand{\caption}[2][\relax]{% Make a new \caption
       {\raggedright\textbf{\ALG@name~\thealgorithm} ##2\par}%
       \ifx\relax##1\relax % #1 is \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##2}%
       \else % #1 is not \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##1}%
       \fi
       \kern2pt\hrule\kern2pt
     }
  }{% \end{breakablealgorithm}
     \kern2pt\hrule\relax% \@fs@post for \@fs@ruled
   \end{center}
  }
\makeatother

\setcounter{secnumdepth}{4}
\setlength{\parskip}{0.5em}
\geometry{a4paper,scale=0.85}  
\title{\textbf{}} 
\author{}  
\date{}
\begin{document}
\maketitle

...

% method 1
\begin{thebibliography}{99}
  \bibitem{ref1} 
\end{thebibliography}
% method 2
\bibliographystyle{plain}
\bibliography{ref}  % add a ref.bib file

\end{document}
```

## Sections

```latex
\section{}
\subsection{}
\subsubsection{}
```

## Enumerate

```latex
\begin{enumerate}
  \item
\end{enumerate}

\begin{itemize}
  \item
\end{itemize}
```

## Table

```latex
\begin{table}[H]
  \centering
  \caption{}
  \begin{threeparttable}
    \begin{tabular}{l||cccc}
      \toprule[2pt]
       &  &  &  &  \\
      \midrule[1pt]
      &  &  &  &  \\
      \bottomrule[2pt]
    \end{tabular}
    \begin{tablenotes}
      \footnotesize
      \item[1] 
    \end{tablenotes}
  \end{threeparttable}
  \label{}
\end{table}
```

## Figure

```latex
%上下并排放置两张图片
\begin{figure}
  \begin{subfigure}{\linewidth}
    \centering
    \includegraphics[width=0.8\linewidth]{figure1}
    \caption{Caption for Figure 1}
    \label{fig:figure1}
  \end{subfigure}

  \bigskip

  \begin{subfigure}{\linewidth}
    \centering
    \includegraphics[width=0.8\linewidth]{figure2}
    \caption{Caption for Figure 2}
    \label{fig:figure2}
  \end{subfigure}

  \caption{Two figures placed vertically}
  \label{fig:combined}
\end{figure}

% 占用双栏
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{}
  \caption{}
  \label{}
\end{figure*}
```

## Algorithm

```latex
\begin{algorithm}[H]
  \caption{}
  \label{}
  \begin{algorithmic}[1]
    \Require 
    \Ensure
    \State 
  \end{algorithmic}
\end{algorithm}
```

## Equation

```latex
\begin{equation}
  \begin{aligned}
    & \\
    &
  \end{aligned}
  \label{}
\end{equation}
```
