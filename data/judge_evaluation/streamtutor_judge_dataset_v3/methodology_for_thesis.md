# Methodology for constructing the StreamTutor judge meta-evaluation dataset

## 1. Research question and scope

The dataset is designed to answer a narrow meta-evaluation question: **which locally deployable judge model most reliably reproduces expert educational-quality judgments for StreamTutor responses?** It is not a general benchmark of all LLM capabilities.

The evaluated object is a tutor response produced in a multi-turn, retrieval-augmented, spoken interaction. Consequently, isolated factual questions are insufficient: the dataset must also test conversation history, elliptical follow-ups, deictic expressions (for example, "this formula"), disfluencies, ASR-like wording, incomplete or irrelevant retrieval, calculations, teaching behavior, and prompt-injection resistance.

## 2. Why 30 questions were not retained

There is no universal minimum sample size that makes an LLM-judge study valid. Thirty isolated questions, however, were inadequate for this thesis because they could not simultaneously support seven course areas, multiple question types, multi-turn behavior, retrieval failure modes, and controlled answer-quality levels.

The redesigned dataset contains 48 unique question turns and 144 scored responses. For intuition, if an observed agreement rate is 0.80, a simple binomial 95% interval has an approximate half-width of 0.143 at n=30, 0.101 at n=60, 0.075 at n=108, and 0.065 at n=144. Kappa uncertainty can be wider when labels are imbalanced; therefore final uncertainty must be estimated by stratified bootstrap rather than relying on these binomial approximations. The dataset is adequate for a master's-level comparative study, not for a universal claim that one judge is best in every domain.

## 3. Evidence base

The design follows these findings:

1. MT-Bench demonstrated the value of multi-turn questions and documented position, verbosity, and self-enhancement biases in LLM judges.
2. Prometheus showed that explicit score rubrics and reference answers materially support fine-grained evaluation and emphasized realistic instructions and balanced score distributions.
3. FLASK showed that decomposed, skill-specific evaluation is more interpretable than one overall score.
4. G-Eval showed that structured evaluation procedures can improve alignment with human judgments while also warning about evaluator preference for LLM-generated text.
5. mtRAG demonstrated that later turns, non-standalone questions, and unanswerable questions are distinct challenges for conversational RAG.
6. RAGTruth showed that response-level and fine-grained grounding annotation are necessary because RAG outputs can remain unsupported or contradictory.
7. LLMBar showed that fluent or engaging but instruction-violating answers can mislead judges; therefore controlled adversarial responses are included.
8. Human-evaluation best-practice research recommends separate criteria, explicit definitions, multiple annotators, agreement with uncertainty, and counterbalanced presentation.

## 4. Source corpus and provenance

Eight instructor-provided slide decks constitute the authoritative course corpus. Each dataset turn records the deck, physical PDF page number, and a short evidence summary. The source manifest records SHA-256 hashes so that the exact source revision is recoverable. The decks themselves are not placed in the dataset archive.

## 5. Sampling design

Sixteen three-turn conversations were authored across seven course areas:

- classification foundations;
- decision trees;
- Naive Bayes;
- clustering and silhouette evaluation;
- feature selection;
- classifier-evaluation basics;
- model comparison and statistical testing.

The sampling is purposive and stratified rather than random. It targets pedagogically important concepts and known system failure modes. Question forms include definitions, explanations, comparisons, procedures, calculations, formula interpretation, applications, misconception correction, and statistical reasoning.

Natural conversational forms include pronouns, ellipsis, topic continuity, self-correction, fillers, ASR-like homophones, and slide-deictic expressions. Every natural utterance is paired with a canonical question, which permits analysis of failures caused by conversational form rather than course content.

## 6. Controlled response construction

Every question has three response variants:

- **excellent**: correct, complete, course-grounded, conversationally coherent, and pedagogically suitable;
- **borderline**: retains the central idea but contains one controlled omission, imprecision, clarity problem, pedagogical weakness, or minor context failure;
- **critical**: contains a major factual/reasoning error, context-resolution failure, fabricated support, non-answer, or prompt-injection compliance.

This counterfactual construction holds the input constant and varies answer quality. It therefore measures whether a judge discriminates substantive quality rather than merely topic, question difficulty, or response length. The variants are not intended to imitate the output distribution of one tutor model; future analyses should additionally report performance on naturally generated tutor responses.

Fourteen additional controlled contrast pairs isolate individual dimensions (two pairs each for correctness, groundedness, completeness, clarity, pedagogy, learner appropriateness, and conversational context). Pairwise order must be reversed, and only position-consistent preferences count. These cases diagnose criterion conflation and superficial-style bias; they are not mixed into the ordinal agreement calculation.

## 7. Rubric revision

The earlier six answer dimensions were a sound foundation but incomplete for a multi-turn voice tutor. Version 3 retains correctness, groundedness, completeness, clarity, pedagogy, and learner appropriateness, and adds **conversational context handling**. Retrieval relevance remains a separate pipeline dimension.

Important separations are enforced:

- correctness is truth relative to the course/reference answer;
- groundedness is support from usable retrieved evidence;
- retrieval relevance measures whether retrieval supplied useful evidence;
- conversational context handling measures pronoun, ellipsis, correction, and history resolution.

Groundedness is N/A when no usable evidence is supplied. Retrieval failure must not be silently converted into a low tutor-groundedness score. Each dimension has behaviorally specific anchors from 1 to 5. Critical errors are recorded separately and can force revision independently of average-looking scores.

Because the controlled excellent/borderline/critical variants are designed around realistic failure profiles, their provisional dimension scores are not artificially uniform. A separate 35-case anchor pack supplies balanced 1--5 examples for all seven answer dimensions. It is used only for rater training and parser/prompt calibration, never for confirmatory judge ranking.

## 8. Splits and leakage control

Whole conversations, rather than individual turns, are assigned to one split:

- calibration: prompt/rubric debugging only;
- selection: frozen confirmatory model comparison;
- robustness: stress tests and failure analysis.

This prevents previous turns from appearing in calibration while their follow-ups appear in selection. Candidate judge prompts, parsing rules, thresholds, and model list must be frozen before inspecting selection results.

## 9. Annotation protocol

The distributed annotations are provisional AI expert labels. They include dimension scores, rationales, critical errors, confidence, and adjudication priority. They are useful for software validation and for creating a rater-training pack, but they are not human ground truth.

The recommended future human protocol is:

1. recruit at least two raters with Data Mining competence;
2. train them on calibration cases, discuss anchor interpretation, and revise only the annotation manual—not selection labels;
3. blind raters to model identity and answer-quality variant;
4. randomize response order while keeping necessary conversation history visible;
5. rate dimensions independently rather than assigning one holistic score;
6. collect confidence and free-text rationale for scores 1, 2, and 5;
7. calculate percentage agreement and quadratic-weighted kappa with stratified bootstrap confidence intervals;
8. adjudicate disagreements larger than one scale point without deleting the original ratings.

If raters remain unavailable, the thesis may report an **AI-provisional engineering comparison**, but it must not claim human agreement, human validation, or identification of the best human-aligned judge.

## 10. Statistical analysis plan

Primary metric: macro-average quadratic-weighted kappa between each judge and adjudicated human ratings across applicable dimensions.

Secondary metrics: per-dimension weighted kappa, mean absolute error, exact agreement, agreement within one point, critical-error precision/recall, invalid-output rate, retry rate, and verdict accuracy.

Use 10,000 stratified bootstrap resamples at the conversation level. Conversation-level resampling preserves dependency among turns and answer variants. When comparing two judges, bootstrap the paired difference on the same resampled conversations. Treat a kappa difference of at most 0.05, or overlapping uncertainty intervals, as practically tied and apply the preregistered tie-break rules.

## 11. Limitations

- Provisional annotations come from an AI system and may share biases with candidate judges.
- The provisional score distribution is design-driven rather than population-representative; agreement metrics must therefore be reported per dimension with uncertainty, and the anchor pack must not be mixed into confirmatory estimates.
- Controlled degradations improve causal interpretability but do not replace naturally occurring tutor errors.
- Forty-eight question turns cannot represent all of Data Mining.
- Course-specific results do not justify claims about other subjects, institutions, languages, or learner populations.
- Some questions depend on visual slide context represented textually; future work should add direct multimodal evaluation if the production tutor sees images.
- Human validation is still required for confirmatory claims.

## 12. Thesis-ready construction statement

"We constructed a course-grounded, conversation-level meta-evaluation dataset from eight authoritative Data Mining slide decks. Sixteen three-turn conversations were purposively stratified across seven curricular units and eleven broad question families. Each of the 48 question turns was paired with an excellent, borderline, and critically defective response, yielding 144 response-evaluation cases. Fourteen additional position-reversed contrast pairs isolated individual rubric dimensions. Controlled variants held question, history, retrieved evidence, learner level, and teaching style constant while varying response quality. Entire conversations were assigned to calibration, frozen selection, or robustness splits to prevent history leakage. Responses were scored independently on correctness, groundedness, completeness, clarity, pedagogy, learner appropriateness, and conversational-context handling; retrieval relevance was evaluated separately. Initial labels were AI-generated provisional expert annotations and were not treated as human ground truth. The confirmatory protocol requires blinded independent human annotation and conversation-level bootstrap confidence intervals."
