"""
Phonetic Similarity vs Inter-word Response Time in Hindi VFT
============================================================

Tests whether consecutive words that are phonetically similar have shorter
inter-word response times (IRTs) than phonetically dissimilar consecutive words.

Three operationalisations of "phonetically similar":
    1. Shared onset consonant
    2. Shared rhyme (last akshara)
    3. Continuous akshara edit similarity, tested both as Spearman ρ and as
       a binary "any akshara overlap vs none" Mann-Whitney contrast.

Tests used:
    - Mann-Whitney U (one-sided, shared < different): non-parametric, robust
      to the right-skew of IRT.
    - Spearman ρ: monotone rank correlation for the continuous measure.
    - Wilcoxon signed-rank paired test: within-participant robustness check
      for the onset contrast (respects nested structure of pairs within subjects).
    - Cliff's δ: non-parametric effect size for the onset contrast.

Input:
    final_vft_data.csv  with columns:
        participant_id, domain, word_order, word, IRT, devnagri_word

Outputs:
    - pairs.csv           consecutive-pair table with all similarity variables
    - per_subj_onset.csv  per-participant mean IRT by shared_onset
    - phonetic_vs_irt.png three-panel box plot figure
    - Console: full set of test results (pooled + per-domain)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Devanagari phonetic-similarity helpers
# ---------------------------------------------------------------------------

VIRAMA = "\u094D"
NUKTA = "\u093C"

# Consonant ranges: base consonants U+0915..U+0939 plus nukta-variants U+0958..U+095F
_CONSONANT_RANGES = [("\u0915", "\u0939"), ("\u0958", "\u095F")]
# Independent vowels U+0905..U+0914
_VOWEL_RANGE = ("\u0905", "\u0914")
# Matras (dependent vowel signs) U+093E..U+094C plus the long vocalic ones U+0962..U+0963
_MATRA_RANGE_1 = ("\u093E", "\u094C")
_MATRA_RANGE_2 = ("\u0962", "\u0963")


def is_consonant(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in _CONSONANT_RANGES)


def is_vowel(ch: str) -> bool:
    return _VOWEL_RANGE[0] <= ch <= _VOWEL_RANGE[1]


def is_matra(ch: str) -> bool:
    return (_MATRA_RANGE_1[0] <= ch <= _MATRA_RANGE_1[1]
            or _MATRA_RANGE_2[0] <= ch <= _MATRA_RANGE_2[1])


def aksharas(word: str) -> list[str]:
    """Split a Devanagari word into aksharas (orthographic syllables).

    Each consonant starts a new akshara and pulls along its matra, nukta,
    and virama. Independent vowels also form their own akshara.
    """
    word = word.strip()
    chunks: list[str] = []
    current = ""
    for ch in word:
        if is_consonant(ch) or is_vowel(ch):
            if current and not current.endswith(VIRAMA):
                chunks.append(current)
                current = ""
            current += ch
        else:
            # Matras, virama, nukta attach to the current akshara
            current += ch
    if current:
        chunks.append(current)
    return chunks


def onset_consonant(word: str) -> str | None:
    """Return the first consonant (or initial vowel) of the word."""
    for ch in word:
        if is_consonant(ch) or is_vowel(ch):
            return ch
    return None


def rhyme(word: str) -> str | None:
    """Return the last akshara of the word (approximate rhyme)."""
    aks = aksharas(word)
    return aks[-1] if aks else None


def edit_distance(seq1, seq2) -> int:
    """Standard Levenshtein distance on arbitrary sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def akshara_edit_similarity(w1: str, w2: str) -> float:
    """1 - (edit distance / max length), both on akshara sequences.
    Returns 0 for empty words.
    """
    a1, a2 = aksharas(w1), aksharas(w2)
    if not a1 or not a2:
        return 0.0
    return 1 - edit_distance(a1, a2) / max(len(a1), len(a2))


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_pair_table(vft: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive-word pairs within each (participant, domain)."""
    vft = vft.sort_values(["participant_id", "domain", "word_order"]).reset_index(drop=True)
    rows = []
    for (pid, dom), grp in vft.groupby(["participant_id", "domain"]):
        grp = grp.sort_values("word_order").reset_index(drop=True)
        for i in range(1, len(grp)):
            w_prev = str(grp.loc[i - 1, "devnagri_word"])
            w_cur = str(grp.loc[i, "devnagri_word"])
            irt = grp.loc[i, "IRT"]
            o_prev = onset_consonant(w_prev)
            o_cur = onset_consonant(w_cur)
            r_prev = rhyme(w_prev)
            r_cur = rhyme(w_cur)
            sim = akshara_edit_similarity(w_prev, w_cur)
            rows.append({
                "participant_id": pid,
                "domain": dom,
                "w_prev": w_prev,
                "w_cur": w_cur,
                "IRT": irt,
                "shared_onset": int(o_prev is not None and o_prev == o_cur),
                "shared_rhyme": int(r_prev is not None and r_prev == r_cur),
                "edit_sim": sim,
                "any_overlap": int(sim > 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def mw_shared_less_than_different(values_shared, values_different):
    """One-sided Mann-Whitney U: is `shared` stochastically less than `different`?"""
    if len(values_shared) < 3 or len(values_different) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(values_shared, values_different, alternative="less")
    return u, p


def cliffs_delta(a, b) -> float:
    """Cliff's delta in [-1, 1]. Computed from the two-sided Mann-Whitney U."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (2 * u) / (len(a) * len(b)) - 1


def format_pair_test(label, shared, different, extra=""):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Shared:     n={len(shared):4d}  mean={shared.mean():8.1f} ms  median={shared.median():8.1f} ms")
    print(f"  Different:  n={len(different):4d}  mean={different.mean():8.1f} ms  median={different.median():8.1f} ms")
    u, p = mw_shared_less_than_different(shared, different)
    if np.isnan(p):
        print("  Mann-Whitney: insufficient data")
    else:
        print(f"  Mann-Whitney U (one-sided, shared < different): U={u:.0f}, p={p:.4g}")
    if extra:
        print(f"  {extra}")


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def pooled_analysis(pairs: pd.DataFrame) -> None:
    print("=" * 72)
    print("POOLED ANALYSIS (all 610 consecutive pairs)")
    print("=" * 72)

    # H1a: shared onset
    shared = pairs.loc[pairs["shared_onset"] == 1, "IRT"]
    diff = pairs.loc[pairs["shared_onset"] == 0, "IRT"]
    delta = cliffs_delta(shared.values, diff.values)
    format_pair_test(
        "H1a: Shared ONSET consonant -> shorter IRT?",
        shared, diff,
        extra=f"Cliff's delta = {delta:+.3f}  (|d|<.147 negligible, <.33 small, <.474 medium)",
    )

    # H1b: shared rhyme
    shared = pairs.loc[pairs["shared_rhyme"] == 1, "IRT"]
    diff = pairs.loc[pairs["shared_rhyme"] == 0, "IRT"]
    format_pair_test("H1b: Shared RHYME (last akshara) -> shorter IRT?", shared, diff)

    # H1c (binary): any akshara overlap
    shared = pairs.loc[pairs["any_overlap"] == 1, "IRT"]
    diff = pairs.loc[pairs["any_overlap"] == 0, "IRT"]
    format_pair_test(
        "H1c (binary): Any akshara overlap (edit_sim > 0) -> shorter IRT?",
        shared, diff,
    )

    # H1c (continuous): Spearman correlation
    print("\nH1c (continuous): Spearman rank correlation(edit_sim, IRT)")
    print("-" * 58)
    rho, p_rho = stats.spearmanr(pairs["edit_sim"], pairs["IRT"])
    print(f"  Spearman rho = {rho:+.4f}, p = {p_rho:.4g}  (negative supports H1)")

    # Within-participant paired test for onset
    print("\nWithin-participant paired test (onset)")
    print("-" * 39)
    per_subj = (pairs.groupby(["participant_id", "shared_onset"])["IRT"]
                .mean().unstack("shared_onset").rename(columns={0: "diff_onset", 1: "same_onset"})
                .dropna())
    print(f"  Participants with both categories: {len(per_subj)}")
    if len(per_subj) >= 5:
        print(f"  Mean of per-subject 'same onset' IRTs: {per_subj['same_onset'].mean():.1f} ms")
        print(f"  Mean of per-subject 'diff onset' IRTs: {per_subj['diff_onset'].mean():.1f} ms")
        w, p_w = stats.wilcoxon(per_subj["same_onset"], per_subj["diff_onset"],
                                alternative="less")
        print(f"  Wilcoxon signed-rank (same < diff, one-sided): W={w:.1f}, p={p_w:.4g}")
    else:
        print("  Not enough paired participants to test.")

    return per_subj


def per_domain_analysis(pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("PER-DOMAIN BREAKDOWN")
    print("=" * 72)

    header = f"{'Domain':<12} {'N':>4}   {'onset_p':>7}  {'rhyme_p':>7}  {'overlap_p':>9}  {'spearman_rho':>12} {'p':>7}"
    print(header)
    print("-" * len(header))

    for dom, g in pairs.groupby("domain"):
        p_onset = mw_shared_less_than_different(
            g.loc[g["shared_onset"] == 1, "IRT"],
            g.loc[g["shared_onset"] == 0, "IRT"],
        )[1]
        p_rhyme = mw_shared_less_than_different(
            g.loc[g["shared_rhyme"] == 1, "IRT"],
            g.loc[g["shared_rhyme"] == 0, "IRT"],
        )[1]
        p_overlap = mw_shared_less_than_different(
            g.loc[g["any_overlap"] == 1, "IRT"],
            g.loc[g["any_overlap"] == 0, "IRT"],
        )[1]
        rho, p_rho = stats.spearmanr(g["edit_sim"], g["IRT"])

        fp = lambda x: f"{x:.3f}" if not np.isnan(x) else "  -  "
        print(f"{dom:<12} {len(g):>4}   {fp(p_onset):>7}  {fp(p_rhyme):>7}  {fp(p_overlap):>9}  "
              f"{rho:>+12.3f} {p_rho:>7.3f}")


def make_figure(pairs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    configs = [
        ("shared_onset", "Shared first consonant",
         ["Different\nonset", "Shared\nonset"]),
        ("shared_rhyme", "Shared last akshara (rhyme)",
         ["Different\nrhyme", "Shared\nrhyme"]),
        ("any_overlap", "Continuous similarity (binarised)",
         ["No\noverlap", "Any akshara\noverlap"]),
    ]

    for a, (col, title, labels) in zip(ax, configs):
        d0 = pairs.loc[pairs[col] == 0, "IRT"].values / 1000
        d1 = pairs.loc[pairs[col] == 1, "IRT"].values / 1000
        a.boxplot([d0, d1], tick_labels=labels, showfliers=False)
        a.set_ylabel("IRT (s)")
        a.set_title(title)
        _, p = mw_shared_less_than_different(pairs.loc[pairs[col] == 1, "IRT"],
                                             pairs.loc[pairs[col] == 0, "IRT"])
        p_str = f"p={p:.3f}" if not np.isnan(p) else "p=n/a"
        a.text(0.02, 0.95, f"n={len(d0)} vs {len(d1)}\nMW {p_str}",
               transform=a.transAxes, ha="left", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vft", default="final_vft_data.csv",
                        help="Path to the VFT CSV file")
    parser.add_argument("--outdir", default=".",
                        help="Directory to write pairs.csv, figure, etc.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vft = pd.read_csv(args.vft)
    print(f"Loaded {len(vft)} rows, {vft['participant_id'].nunique()} participants, "
          f"{vft['domain'].nunique()} domains: {sorted(vft['domain'].unique())}")

    pairs = build_pair_table(vft)
    pairs.to_csv(outdir / "pairs.csv", index=False)
    print(f"Built {len(pairs)} consecutive pairs "
          f"(shared_onset={pairs['shared_onset'].sum()}, "
          f"shared_rhyme={pairs['shared_rhyme'].sum()}, "
          f"any_overlap={pairs['any_overlap'].sum()})")

    per_subj = pooled_analysis(pairs)
    if per_subj is not None:
        per_subj.to_csv(outdir / "per_subj_onset.csv")

    per_domain_analysis(pairs)
    make_figure(pairs, outdir / "phonetic_vs_irt.png")


if __name__ == "__main__":
    main()
