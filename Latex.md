# Latex Grammer

- [Latex Grammer](#latex-grammer)
  - [Mathematical Symble](#mathematical-symble)
  - [Title \& Packages](#title--packages)
  - [Footnote](#footnote)
  - [Sections](#sections)
  - [Enumerate](#enumerate)
  - [listings](#listings)
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
\usepackage{enumitem}
\usepackage{geometry}
\usepackage{tikz}
\usepackage{verbatim}
\usepackage{amsmath, bm}
\usepackage{mathrsfs}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}
\usepackage{wrapfig}
\usepackage[colorlinks,
            linkcolor=blue,
            anchorcolor=blue,
            citecolor=blue]{hyperref}
\usepackage{caption}
\captionsetup[figure]{font=small, labelfont={bf,it}, name={图}}
\captionsetup[table]{font=small, labelfont={bf,it}, name={表}}

\setcounter{secnumdepth}{4}
\setlength{\parskip}{0.3em}
\geometry{a4paper,scale=0.8}  
\title{\textbf{}} 
\author{}  
\date{}
\begin{document}
\maketitle

...

% reference
% method 1
\begin{thebibliography}{99}
  \bibitem{ref1} ...
  \bibitem{ref2} ...
\end{thebibliography}
% method 2
\bibliographystyle{plain}
\bibliography{ref}  % add a ref.bib file

\end{document}
```

## Footnote
```latex
....\footnotemark[1]....

\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\footnotetext[1]{......}
```

## Sections
```latex
\section{}
\subsection{}
\subsubsection{}
```

## Enumerate
```latex
\begin{enumerate}[leftmargin=*]
  \item
\end{enumerate}

\begin{itemize}[leftmargin=*]
  \item[a)]
\end{itemize}
```

## listings
```latex
\usepackage{listings}

\setmonofont{Menlo}[Scale=0.9]
\definecolor{vscode-keyword}{HTML}{569CD6}
\definecolor{vscode-string}{HTML}{CE9178}
\definecolor{vscode-comment}{HTML}{6A9955}
\definecolor{vscode-bg}{HTML}{F5F5F5}
\lstset{frame=tb,
  language=python,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=flexible,
  basicstyle={\small\ttfamily},
  numbers=left,
  numbersep=1pt,
  keywordstyle=\color{vscode-keyword}\bfseries,
  commentstyle=\color{vscode-comment}\itshape,
  stringstyle=\color{vscode-string},
  numberstyle=\tiny\color{vscode-comment},
  backgroundcolor=\color{vscode-bg},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=4
}
\renewcommand{\lstlistingname}{Code}
\captionsetup[lstlisting]{labelfont=bf}

\begin{lstlisting}[language=Python, caption={xxx}, label={lst:xxx}]
  ...
\end{lstlisting}
```

## Table
```latex
\usepackage{multirow}
\usepackage{threeparttable}
\usepackage{booktabs}

%simple table
\begin{table}[htbp]
  \centering
  \caption{}
  \renewcommand\arraystretch{1.25}
  \resizebox{\linewidth}{!}{  % resize table
    \begin{tabular}{l|cccc}
      \bottomrule[2pt]
      \multirow{2}{*}{Multirow} & \multicolumn{2}{c|}{Multicolumn1} & \multicolumn{2}{c|}{Multicolumn2}  \\
      \cline{2-5}
      ~ & 1.1 & 1.2 & 2.1 & 2.2 \\
      \hline
      &  &  &  &  \\
      \toprule[2pt]
    \end{tabular}
  }
  \label{}
\end{table}

%table notes with threeparttable
\begin{table}[htbp]
  \centering
  \caption{}
  \begin{threeparttable}
  \resizebox{\linewidth}{!}{  % resize table
    \begin{tabular}{l|cccc}
      \toprule[2pt]
       &  &  &  &  \\
      \midrule[1pt]
      &  &  &  &  \\
      \bottomrule[2pt]
    \end{tabular}
  }
    \begin{tablenotes}
      \footnotesize
      \item[1] 
    \end{tablenotes}
  \end{threeparttable}
  \label{}
\end{table}

%table notes without threeparttable
\begin{table}[htbp]
  \centering
  \caption{}
  \resizebox{\linewidth}{!}{  % resize table
    \begin{tabular}{l|cccc}
      \toprule[2pt]
       &  &  &  &  \\
      \midrule[1pt]
      &  &  &  &  \\
      \bottomrule[2pt]
    \end{tabular}
  }
  \noindent
  \begin{minipage}{\linewidth}
    \hspace{0.1cm}
    \vspace{-0.1cm}
    \scriptsize
    \begin{itemize}[leftmargin=*]
      \item[1] 
    \end{itemize}
  \end{minipage}
  \label{}
\end{table}
```

## Figure
```latex
%上下并排放置两张图片
\begin{figure}[htbp]
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

%占用双栏
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{}
  \caption{}
  \label{}
\end{figure*}
```

## Algorithm
```latex
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}

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

\begin{breakablealgorithm}[htbp]
  \caption{}
  \label{}
  \begin{algorithmic}[1]
    \Require 
    \Ensure
    \State 
  \end{algorithmic}
\end{breakablealgorithm}
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

