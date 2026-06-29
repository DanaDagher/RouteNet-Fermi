"""
md_to_tex.py — pragmatic Markdown -> LaTeX fragment converter for the
deliverable. Produces a \section-level fragment (no preamble) to paste/\input
into an existing LaTeX document.

Required LaTeX packages in the host doc: booktabs, graphicx, amsmath, listings,
hyperref. NOTE: a few inline equations (delay-mechanism section) may need manual
LaTeX touch-up — proof-read the output.
"""
import re, sys, os

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/deliverable_subsections_DRAFT.md"
DST = sys.argv[2] if len(sys.argv) > 2 else "results/deliverable_subsections.tex"

UNI = {
    "≈": r"$\approx$", "×": r"$\times$", "−": r"$-$", "→": r"$\rightarrow$",
    "≡": r"$\equiv$", "≤": r"$\leq$", "≥": r"$\geq$", "ρ": r"$\rho$",
    "σ": r"$\sigma$", "λ": r"$\lambda$", "Σ": r"$\Sigma$", "π": r"$\pi$",
    "…": r"\ldots{}", "•": r"\textbullet{}", "✓": r"\checkmark{}",
    "✔": r"\checkmark{}", "✅": r"\checkmark{}", "⚠": r"$\triangle$",
    "≪": r"$\ll$", "≫": r"$\gg$", "—": "---", "–": "--",
    "’": "'", "‘": "'", "“": "``", "”": "''", "²": r"$^{2}$",
    "⁻⁴": r"$^{-4}$", "⁻⁵": r"$^{-5}$", "⁻": r"$^{-}$", "̂": "",
}
SUP = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}


def esc(t):
    # protect inline code first
    codes = []
    def stash(m):
        codes.append(m.group(1)); return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", stash, t)
    # escape latex specials
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    # unicode
    for a, b in UNI.items():
        t = t.replace(a, b)
    for a, b in SUP.items():
        t = t.replace(a, "$^{"+b+"}$")
    # bold / italic
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", t)
    # links [text](url) -> text (footnote url optional)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    # restore code
    def unstash(m):
        c = codes[int(m.group(1))]
        c = c.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
        return r"\texttt{" + c + "}"
    t = re.sub(r"\x00(\d+)\x00", unstash, t)
    return t


def cols(n):
    return "l" * n if n <= 3 else "l" + "p{0.30\\textwidth}" * (n - 1)


def preprocess(lines):
    """Merge soft-wrapped continuation lines into one logical line per block
    (outside code fences/tables), so multi-line bold and wrapped list items work."""
    res, in_code = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_code = not in_code; res.append(ln); continue
        if in_code:
            res.append(ln); continue
        is_struct = (s == "" or s == "---" or s.startswith("#") or s.startswith("|")
                     or s.startswith("![") or s.startswith(">")
                     or re.match(r"^([-*]|\d+\.)\s+", s) is not None)
        if is_struct or not res:
            res.append(ln); continue
        prev = res[-1].strip()
        prev_struct = (prev == "" or prev == "---" or prev.startswith("#")
                       or prev.startswith("|") or prev.startswith("![") or prev.startswith(">"))
        if prev_struct:
            res.append(ln)
        else:
            res[-1] = res[-1].rstrip() + " " + s   # continuation -> merge
    return res


def main():
    lines = preprocess(open(SRC, encoding="utf-8").read().split("\n"))
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        # code fence
        if ln.strip().startswith("```"):
            out.append(r"\begin{lstlisting}")
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                out.append(lines[i]); i += 1
            out.append(r"\end{lstlisting}"); i += 1; continue
        # heading
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl, txt = len(m.group(1)), esc(m.group(2))
            cmd = {1: r"\section*", 2: r"\section", 3: r"\subsection", 4: r"\subsubsection"}[lvl]
            out.append(f"{cmd}{{{txt}}}"); i += 1; continue
        # image -> figure (consume following *Figure ...* caption if present)
        mi = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln)
        if mi:
            path = mi.group(2); cap = mi.group(1)
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip().startswith("*Figure"):
                cap = esc(lines[j].strip().strip("*")); i = j
            out += [r"\begin{figure}[h]", r"\centering",
                    rf"\includegraphics[width=0.6\textwidth]{{{path}}}",
                    rf"\caption{{{cap}}}", r"\end{figure}"]
            i += 1; continue
        # table
        if ln.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i+1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            body = []
            while i < n and lines[i].strip().startswith("|"):
                body.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            ncol = len(header)
            out += [r"\begin{table}[h]", r"\centering", rf"\begin{{tabular}}{{{cols(ncol)}}}", r"\toprule"]
            out.append(" & ".join(esc(c) for c in header) + r" \\")
            out.append(r"\midrule")
            for row in body:
                row += [""] * (ncol - len(row))
                out.append(" & ".join(esc(c) for c in row[:ncol]) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            continue
        # horizontal rule
        if ln.strip() == "---":
            out.append(r"\bigskip"); i += 1; continue
        # list block
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            env = "enumerate" if re.match(r"^\s*\d+\.\s+", ln) else "itemize"
            out.append(rf"\begin{{{env}}}")
            while i < n and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i])
                out.append(r"\item " + esc(item)); i += 1
            out.append(rf"\end{{{env}}}"); continue
        # blank
        if ln.strip() == "":
            out.append(""); i += 1; continue
        # paragraph
        out.append(esc(ln)); i += 1

    open(DST, "w", encoding="utf-8").write("\n".join(out))
    print("wrote", DST, os.path.getsize(DST), "bytes")


if __name__ == "__main__":
    main()
