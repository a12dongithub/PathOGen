# WACV 2027 Author Kit (Latex)

### Command-line (e.g. Linux)
If you plan to generate your paper via the command-line, you may need to install latex packages/fonts (e.g. `texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended`).

To generate the document from the latex file, it is recommended that you use `pdflatex`.  For example, also using `bibtex` to generate references:

```$ pdflatex main.tex; bibtex main; pdflatex main.tex```

### Microsoft Word Template (None Provided)
We are no longer providing a Microsoft Word template, and are actively discouraging the use of Word for preparing papers for WACV.  Latex is used for the vast majority of conferences and journals in our field and Overleaf makes the use of Latex very straightforward, even for first-time users.  Overleaf provides an excellent tutorial titled [Learn Latex in 30 minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes).


## Notes on Review vs. Camera-ready Formats:
The typical `egpaper_for_review.tex` and `egpaper_final.tex` files have been merged into a single file, `main.tex`.  The file is initially set up for review submission -- all you need to add is your Paper ID and select your track (Applications or Algorithms).

There are important instructions at the top of the combined `main.tex` document describing (i) how to toggle between the review and final (camera-ready) formats, (ii) how to set your Paper ID (which is assigned upon creation of your paper submission in CMT) for the review version, and (iii) how to select your track (Applications or Algorithms) in the review copy.

### History (in reverse chronological order)

- updated for WACV 2027, adding dataset track, by [Christopher Funk](https://www.kitware.com/christopher-funk/))
- updated for WACV 2026, merging WACV-specific aspects (e.g., two tracks) from the WACV 2025 template into the ICCV 2025 template, by [Vlad Morariu](https://openreview.net/profile?id=~Vlad_I_Morariu1)
- added styles for `subsubsection` and fixed the wrong PDF bookmarks by [Di Fang](https://github.com/fang-d)
- modernized for CVPR 2025 by [Christian Richardt](https://richardt.name/)
- fixed page centering for CVPR 2025 by [Stefan Roth](mailto:stefan.roth@NOSPAMtu-darmstadt.de)
- inline enumerations and `cvprblue` links for CVPR 2025 by [Ioannis Gkioulekas
](https://www.cs.cmu.edu/~igkioule/)
- added automated LaTeX build testing for CVPR 2025 by [Ahan Shabanov](https://ahanio.github.io)
- references in `cvprblue` for CVPR 2024 by [Klaus Greff](https://github.com/Qwlouse) 
- added natbib for CVPR 2024 by [Christian Richardt](https://richardt.name/)
- replaced buggy (review-mode) line numbering for 3DV 2024 by [Adín Ramírez Rivera
](https://openreview.net/profile?id=~Ad%C3%ADn_Ram%C3%ADrez_Rivera1)
- logic for inline supplementary for 3DV 2024 by [Andrea Tagliasacchi](https://taiya.github.io) 
- modernized for CVPR 2022 by [Stefan Roth](mailto:stefan.roth@NOSPAMtu-darmstadt.de)
- created cvpr.sty file to unify review/rebuttal/final versions by [Ming-Ming Cheng](https://github.com/MCG-NKU/CVPR_Template)
- developed CVPR 2005 template  by [Paolo Ienne](mailto:Paolo.Ienne@di.epfl.ch) and [Andrew Fitzgibbon](mailto:awf@acm.org)
