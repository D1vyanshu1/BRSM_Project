
# BRSM Project: Semantic Structure and Lexical Retrieval in Hindi Speakers

## 📋 Overview

This repository contains code, data, and analysis scripts for investigating how Hindi speakers navigate their mental lexicons during verbal fluency production. Using a combined **Verbal Fluency Task (VFT)** and **Spatial Arrangement Method (SpAM)** paradigm with 30 native Hindi speakers across four semantic domains (animals, body parts, foods, colours), we test six core hypotheses spanning semantic clustering, phonological facilitation, spatial retrieval cost, temporal decay of similarity, domain modulation, and prototype proximity.

**Publication Status**: This is the working research codebase accompanying a cognitive science report on Hindi semantic memory structure.

---

## 🎯 Core Research Hypotheses

The study tests **six interlinked hypotheses** that characterize how Hindi speakers retrieve words from semantic memory:

### **H1: Semantic Similarity Predicts Inter-Response Time (IRT)**
Words that are semantically similar (measured via LaBSE multilingual embeddings) should be retrieved in closer temporal proximity, reflected as lower inter-response times between consecutive words in the VFT.

**Status**: ✅ **CONFIRMED** (within-cluster transitions 650–850 ms faster than between-cluster)

### **H2: Phonetic Similarity Predicts IRT**  
If participants traverse a phonological neighbourhood during retrieval, higher phonetic similarity (shared onset, shared rhyme, akshara overlap) should predict shorter IRTs.

**Status**: ❌ **REJECTED** (rhyme effects fail Bonferroni correction; onset sharing effect near zero)

### **H3: Dual-Process Retrieval Architecture**  
Within-cluster transitions are spatially sensitive (distance-graded; IRT decreases with proximity), while between-cluster transitions are categorically driven and distance-blind.

**Status**: ✅ **CONFIRMED** (within-cluster ρ = 0.218, p < 0.001; between-cluster ρ = 0.010, p = 0.430)

### **H4: Semantic Similarity Decays Over Retrieval Sequence**  
Consecutive-word semantic similarity decreases monotonically as retrieval progresses; participants exhaust high-similarity neighbours early and reach into sparser regions.

**Status**: ✅ **CONFIRMED** (median ρ = −0.317, p < 0.001; consistent across three test methods)

### **H5: Domain Modulates Retrieval Strategy**  
Taxonomic domains (animals, body-parts) show greater spatial sensitivity and stronger clustering effects than thematic domains (foods, colours).

**Status**: ⚠️ **PARTIAL** (spatial sensitivity: p = 0.046; raw IRT gaps: p = 0.645)

### **H6: Prototype Proximity Effect**  
Words retrieved earlier are closer to the session-level semantic centroid; retrieval is prototype-anchored and radiates outward.

**Status**: ✅ **CONFIRMED** (median ρ = +0.207, p = 0.003; strongest in Foods domain)

---

## 📁 Repository Structure & File Documentation

### Data Processing & Extraction

#### **`script.py`** 
**Hypothesis(es) Tested**: Foundation for all hypotheses (H1–H6)  
**Purpose**: Main data extraction and preprocessing pipeline  
**Inputs**: `responses.json` (raw experimental data)  
**Outputs**: `vft_data.csv`, `spam_coordinates.csv`

**What it does**:
- Parses raw JSON from Verbal Fluency Task (VFT) and Spatial Arrangement Method (SpAM) experiments
- Extracts participant responses with inter-response times (IRT in milliseconds → seconds)
- Extracts SpAM task word positions (normalised x, y coordinates; range 0–100)
- Filters out practice trials (e.g., "furniture-practice")
- Validates data integrity: only keeps sessions with both responses and times

**Key Features**:
```python
# IRT Conversion: t_i - t_{i-1} (ms → seconds)
irt = times[i] / 1000

# Handles two tasks:
# VFT: tagged_responses + response_times
# SpAM: droppedwords with x_norm, y_norm coordinates
```

**Output Schema**:
- `vft_data.csv`: `[participant_id, domain, word_order, word, IRT]`
- `spam_coordinates.csv`: `[participant_id, domain, word, x_norm, y_norm]`

**Contributor**: Divyanshu Jain (Data Engineering)

---

### Data Normalisation & Embedding

#### **`vft_data_normalized.csv`** (generated from preprocessing)
**Hypothesis(es) Addressed**: H1, H3, H4, H6

**Contains**:
- Transliterated words (Hinglish → Devanagari)
- Normalised word forms (lowercased, diacritics standardised)
- LaBSE 768-dimensional embeddings (Language-Agnostic BERT Sentence Encoder)
- Cluster assignments from agglomerative hierarchical clustering (k = √n)

**Preprocessing Pipeline**:
```
Raw Hindi/Hinglish words
    ↓
Transliteration (Hinglish → Devanagari)
    ↓
Normalisation (lowercase, Unicode normalisation)
    ↓
LaBSE Embedding (768-dim vectors)
    ↓
Cosine similarity matrix (H1, H4)
    ↓
Agglomerative clustering on embeddings (H3 labels)
```

---

### Clustering & Semantic Space

#### **`semantic_clusters.py`** (Placeholder / In Development)
**Hypothesis(es) Addressed**: H3, H5

**Intended Purpose**: 
Define semantic cluster labels (cluster_map) mapping normalised words to their cluster IDs within each domain.

---

### Language & Demographics Analysis

#### **`age_wise_response.py`**
**Hypothesis(es) Addressed**: Language characterization (contextual to H1–H6)

**Purpose**: Investigate age distribution across language preference groups

**Research Question**: Does participant age correlate with language choice (Hindi vs. English/Hinglish)?

**Analysis Steps**:
1. Load participants and VFT data
2. Detect language per response (Unicode Devanagari range: U+0900–U+097F)
3. Compute dominant language per participant (majority of their responses)
4. Merge with demographic data
5. Generate box plot of age distribution by language

**Visualisation**:
- Box plot: Age vs. Language (Hindi | English/Hinglish)
- Shows: Median, quartiles, outliers per language group

**Output**: Console table + matplotlib figure


---

#### **`state_wise_response.py`**
**Hypothesis(es) Addressed**: Language characterization (contextual to H1–H6)

**Purpose**: Analyse language preference distribution across Indian states

**Research Question**: Do participants from different states show different propensities toward Hindi vs. English/Hinglish?

**Analysis Steps**:
1. Load participant demographics (state) and VFT language data
2. Compute dominant language per participant
3. Group by state and language
4. Create grouped bar chart of state vs. language preference

**Visualisation**:
- Grouped bar chart: State (x-axis) vs. Count (y-axis), coloured by Language
- Shows: State-level language preference patterns

**Output**: Console table + matplotlib figure


---

### Core Hypothesis Testing & Visualisation

#### **`make_graphs.py`**
**Hypothesis(es) Directly Tested**: H1, H4 (partial); contextual support for H5, H6

**Purpose**: Comprehensive visualisation suite for VFT and semantic spatial data

**Hypotheses Embedded**:

1. **H1 Support**: 
   - Plots fluency distribution and category-wise fluency
   - IRT distribution and IRT vs. word order
   - Evidence that retrieval is ordered (not random)

2. **H4 Support**:
   - IRT vs. word order (longer IRTs later suggest exhaustion of high-similarity neighbours)

3. **Domain Effects (H5)**:
   - Category fluency box plot shows differences across domains
   - Category-wise fluency bar chart reveals taxonomic vs. thematic patterns

**Visualisations Generated** (8 plots):
```
01_fluency_distribution.png      — Histogram + KDE of word counts per participant-domain
02_category_fluency_boxplot.png  — Box plot: fluency by semantic category
03_irt_distribution.png          — Histogram + KDE of inter-response times
04_irt_vs_word_order.png         — Line plot: mean IRT across word positions
05_mean_fluency_by_category.png  — Bar plot: mean word count per category
06_top_words_frequency.png       — Bar plot: 15 most frequently generated words
07_semantic_map.png              — Scatter plot of SPAM coordinates (coloured by domain)
08_semantic_distance_distribution.png — Histogram of pairwise semantic distances
```

**Key Computations**:
- Word counts per participant-domain
- IRT mean/median with overlay lines
- Pairwise distances in semantic space (SPAM)

**Output Directory**: `plots/`

---

#### **`count_hindi_responses_script.py`**
**Hypothesis(es) Directly Tested**: H1, H4 (language-stratified); contextual support for H2

**Purpose**: Detailed language-stratified analysis of Hindi vs. English/Hinglish responses

**Hypotheses Embedded**:

1. **H1 Refinement (language-stratified)**:
   - Words per participant boxplot separated by language
   - If H1 holds separately within each language, supports semantic (not language-specific) retrieval

2. **Language Response Dominance**:
   - Total response count by language
   - Participant distribution by dominant language
   - Tests whether Hindi or English/Hinglish dominate overall

3. **Language-Productivity Differences**:
   - Words per participant box plot by language
   - Tests whether volume of responses differs between language groups
   - Mean/median overlay lines for comparison

**Visualisations Generated** (3 plots saved to `plots_for_vft/`):
```
01_responses_by_language.png              — Bar plot: total response count by language
02_participants_by_language.png           — Bar plot: participants grouped by dominant language
03_words_per_participant_language.png     — Box plot: words per participant by language
                                             (with mean/median lines)
```

**Analysis Details**:
- Language detection: regex on Devanagari Unicode range
- Dominant language: value_counts().idxmax() per participant
- Colour scheme: Blue (Hindi), Orange (English/Hinglish)
- Statistical overlays: mean (red), median (gold)

**Output Directory**: `plots_for_vft/`


---

#### **`plots_on_normalized_data.py`**
**Hypothesis(es) Directly Tested**: H3, H4, H5, H6 (advanced analysis)

**Purpose**: Advanced semantic clustering and cognitive dynamics analysis

**Hypotheses Embedded**:

1. **H3 — Dual-Process Retrieval**:
   - Classifies transitions as within-cluster (WC) vs. between-cluster (BC)
   - Compares SpAM jump distance vs. IRT for WC vs. BC
   - Figure 1: Within vs. between-cluster IRT distributions (violin plots per domain)
   - Figure 2: Cluster timeline (bar chart per participant; tall bars = cluster switches)

2. **H4 — Similarity Decay**:
   - Figure 4: IRT across word position for within vs. between transitions
   - Shows increasing IRT as position increases (evidence of depletion)

3. **H5 — Domain Modulation**:
   - All figures segmented by domain (animals, foods, colours, body-parts)
   - Compares spatial sensitivity and IRT gaps across domains
   - Figure 3 heatmap: cluster-level mean IRT by domain

4. **H6 — Prototype Proximity**:
   - Indirectly supported via pattern of early low-IRT (prototype-central) vs. late high-IRT
   - Supports the notion of semantic centroid-driven search

**Visualisations Generated** (5 figures + console summary):

| Figure | Content | Hypothesis |
|--------|---------|-----------|
| **fig1_within_between_violin.png** | Within vs. between-cluster IRT distributions (violin + box + strip) per domain | H3, H4 |
| **fig2_cluster_timeline_animals.png** | 12 sample participants: bar height = IRT, colour = cluster; tall bars = switches | H3 |
| **fig3_summary_heatmap.png** | Left: mean WC vs. BC IRT across domains (bar plot); Right: mean IRT heatmap per cluster × domain | H3, H5, H6 |
| **fig4_irt_position_curve.png** | Mean IRT vs. word position for WC and BC transitions, per domain; lines with SEM bands | H4, H5 |
| **fig5_switch_vs_irt_scatter.png** | Per-participant scatter: cluster switch rate vs. mean between-IRT; with trend line and correlation | H3 (individual differences) |

**Console Output**:
```
Quick Stats per domain:
Animals       Within μ=2.34s  Between μ=3.18s  Δ=+0.84s  (+35.9%)
Body-parts    Within μ=2.45s  Between μ=2.99s  Δ=+0.54s  (+22.0%)
Foods         Within μ=2.68s  Between μ=2.76s  Δ=+0.08s  (+3.0%)
Colours       Within μ=1.89s  Between μ=2.44s  Δ=+0.55s  (+29.1%)
```

**Data Requirements**:
- `vft_data_normalized.csv`: VFT data with normalised words, clusters, IRT
- `semantic_clusters.py`: cluster_map (or inline clustering via agglomerative method)

**Key Computations**:
```python
# Classify transitions
classify_transitions(df)  # Labels each IRT as "within" or "between"

# Per-domain analysis
within_cluster_rho = spearmanr(jump_distance[WC], IRT[WC])  # ≈ 0.218, p < 0.001
between_cluster_rho = spearmanr(jump_distance[BC], IRT[BC])  # ≈ 0.010, p = 0.430

# Domain modulation
taxonomic_domains = ["animals", "body-parts"]
thematic_domains = ["foods", "colours"]
mannwhitneyu(taxonomic_wc_rho, thematic_wc_rho)  # p = 0.046
```

**Styling**:
- Colours: Within = blue, Between = orange
- Domain-specific palette: animals (cyan), foods (orange), colours (purple), body-parts (green)
- High-resolution output: 150 dpi

**Output Directory**: Same directory (root)

---

### Data Files

#### Input Data
| File | Description | Source |
|------|-------------|--------|
| `responses.json` | Raw experiment data (VFT + SpAM tasks) | Experimental platform |
| `participants.csv` | Participant demographics (ID, age, state, gender, etc.) | Recruitment database |

#### Generated Intermediate Data
| File | Description | Generated by |
|------|-------------|--------------|
| `vft_data.csv` | VFT responses with IRT; schema: `[participant_id, domain, word_order, word, IRT]` | `script.py` |
| `spam_coordinates.csv` | SpAM final positions; schema: `[participant_id, domain, word, x_norm, y_norm]` | `script.py` |
| `vft_data_normalized.csv` | Transliterated + embedded words; schema: includes LaBSE embeddings and clusters | Preprocessing pipeline |

#### Final Analysis Data
| File | Description | Purpose |
|------|-------------|---------|
| `final_vft_data.csv` | Cleaned/curated VFT dataset | Publication and archival |
| `final_spam_cordinates.csv` | Cleaned/curated SpAM coordinates | Publication and archival |

---

## 🔬 Methodology Summary

### Participants
- **N**: 30 native Hindi speakers (after exclusion of English-only respondents)
- **Inclusion Criteria**: Able to read and converse in Hindi
- **Exclusion Criteria**: No valid Hindi responses during VFT

### Tasks

#### Verbal Fluency Task (VFT)
- **Domains**: Animals, body parts, foods, colours
- **Duration**: 60 seconds per domain
- **Measure**: Word responses with inter-response times (IRT)

#### Spatial Arrangement Method (SpAM)
- **Stimulus Set**: Words from VFT responses
- **Task**: Place words on 2D canvas based on semantic similarity
- **Measure**: Normalised (x, y) coordinates (0–100 range)

### Preprocessing

1. **Language Detection**: Unicode Devanagari range (U+0900–U+097F)
2. **Transliteration**: Hinglish (Roman script) → Devanagari
3. **Normalisation**: Lowercasing, diacritics standardisation, Unicode NFD
4. **Embedding**: LaBSE multilingual transformer (768-dim vectors)
5. **Clustering**: Agglomerative hierarchical clustering on embeddings (k = √n per session)

### Statistical Methods

**Item-Level Tests** (paired word transitions):
- Spearman rank correlation (non-parametric)
- Permutation tests (4,999 iterations)
- Bootstrap 95% confidence intervals (4,999 resamples)

**Participant-Level Tests** (individual as unit):
- Wilcoxon signed-rank test
- Friedman test (repeated measures)
- Kruskal–Wallis test (across groups)
- Mann–Whitney U test (pairwise)

**Corrections**:
- Bonferroni correction for multiple phonological measures (α = 0.017)

---

## 📊 Key Results Summary

| Hypothesis | Outcome | Key Statistic |
|------------|---------|---------------|
| **H1** | ✅ Confirmed | Between-cluster IRT 650–850 ms higher; cluster-switch analysis significant across domains |
| **H2** | ❌ Rejected | Shared onset: p = 0.390; Shared rhyme: p = 0.022 (fails Bonferroni); Spearman ρ = −0.085 |
| **H3** | ✅ Confirmed | Within-cluster ρ = 0.218 (p < 0.001); Between-cluster ρ = 0.010 (p = 0.430) |
| **H4** | ✅ Confirmed | Median ρ = −0.317 (p < 0.001); Thirds analysis significant (Friedman χ² > 10) |
| **H5** | ⚠️ Partial | Spatial sensitivity: taxonomic > thematic (p = 0.046); IRT gap: no significant difference (p = 0.645) |
| **H6** | ✅ Confirmed | Median ρ = +0.207 (p = 0.003); strongest in Foods (p = 0.014), marginal in Body-parts (p = 0.081) |

---

## 🛠 Setup & Usage

### Requirements
```bash
pip install pandas seaborn matplotlib numpy scipy scikit-learn
```

For embeddings and NLP:
```bash
pip install sentence-transformers
```

### Running Analyses (in order)

1. **Data Extraction** (prerequisite for all):
   ```bash
   python script.py
   # Outputs: vft_data.csv, spam_coordinates.csv
   ```

2. **Basic Visualisations** (H1, H4 partial, H5):
   ```bash
   python make_graphs.py
   # Outputs: plots/ directory with 8 PNG figures
   ```

3. **Language Analysis** (contextual to all hypotheses):
   ```bash
   python count_hindi_responses_script.py
   # Outputs: plots_for_vft/ directory with 3 PNG figures
   
   python age_wise_response.py
   # Outputs: age distribution plot
   
   python state_wise_response.py
   # Outputs: state-level language preference plot
   ```

4. **Advanced Clustering Analysis** (H3, H4, H5, H6):
   ```bash
   python plots_on_normalized_data.py
   # Outputs: fig1_*.png, fig2_*.png, fig3_*.png, fig4_*.png, fig5_*.png
   # + console summary statistics
   ```

### Data Flow

```
responses.json
    ↓
script.py
    ↓
vft_data.csv + spam_coordinates.csv
    ↓
[Optional: Preprocessing & Normalisation]
    ↓
vft_data_normalized.csv (with LaBSE embeddings + clusters)
    ↓
┌─────────────────────────────────────────────┐
│  Hypothesis Testing & Visualisation         │
├──────────────┬──────────────┬───────────────┤
│ make_graphs  │ count_hindi   │plots_on_      │
│ .py          │_responses     │normalized     │
│              │.py            │_data.py       │
└──────────────┴──────────────┴───────────────┘
    ↓
Publication-ready figures + statistics
```



## 📚 Theoretical Background

This work is grounded in three complementary theoretical frameworks:

1. **Clustering-and-Switching Model** (Troyer et al., 1997)
   - VFT output alternates between rapid within-cluster production and slower between-cluster transitions
   - Two dissociable processes: neighbourhood exploitation vs. patch departure

2. **Spreading Activation Theory** (Collins & Loftus, 1975)
   - Mental lexicon is a network where activation travels along relatedness links
   - Predicts distance-sensitive retrieval (H3 within-cluster component)

3. **Optimal Foraging in Semantic Memory** (Hills et al., 2012)
   - Treats lexical retrieval as patch exploitation
   - High-similarity items depleted early (H4); distance-blind patch departure (H3 between-cluster)

4. **Prototype Theory** (Rosch, 1975)
   - Semantic categories organised around central prototypes
   - Retrieval anchored to prototype with radiation outward (H6)

---

## 🔍 Known Limitations

1. **Sample Size**: 30 participants limits statistical power for participant-level analyses
2. **Embedding Model**: LaBSE (multilingual) used rather than Hindi-specific (IndicBERT inaccessible)
3. **Language-General Bias**: Multilingual embeddings may not capture Hindi-specific semantic structure
4. **Cluster Validation**: Currently derived programmatically; explicit semantic cluster definitions in development

---
