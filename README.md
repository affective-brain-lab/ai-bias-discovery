# Humans systematically undervalue information under complexity: A novel bias-discovery approach

Moshe Glickman1, 2 & Tali Sharot1, 2, 3

1 Affective Brain Lab, Department of Experimental Psychology, University College London, London, UK <br />
2 Max Planck UCL Centre for Computational Psychiatry and Ageing Research, University College London, London, UK <br />
3 Department of Brain and Cognitive Sciences, Massachusetts Institute of Technology, Cambridge, MA, USA <br />

Correspondence authors: mosheglickman345@gmail.com, t.sharot@ucl.ac.uk

Biases in how people seek and integrate information can lead to negative consequences across domains ranging from health to finance. It is thus crucial to identify common biases. Traditionally, researchers generate hypotheses regarding potential biases and then empirically test them. While successful, important biases may go undetected as researchers have limited cognitive resources and can be blind to their own biases. To overcome this, we developed a novel data-driven framework for bias discovery. We use machine learning to identify behavioral patterns that systematically deviate from a normative benchmark and large language models to interpret these patterns. We validate the identified biases through controlled experiments. Using this approach, we discovered a new bias: undervaluation of information under complexity. Across multiple experiments (N = 444), we reveal that when faced with complex situations, where information is particularly beneficial, participants systematically underestimate its value. In contrast, they overpay for information on simple trials, where information is least useful. As a result, they make worse decisions, resulting in monetary losses. Our AI-assisted method can be used to identify biases in domains ranging from perception to social judgment, significantly advancing the understanding of human cognition. 

## Experiment 1

### Background
Experiment 1 (N = 304) introduced an incentive-compatible information-seeking lottery task. On each trial, participants viewed six cards with positive or negative values (−9 to 9, excluding 0) and bid between 0 and 1.5 points for information that would remove three of the six cards. Bids were elicited using the Becker–DeGroot–Marschak procedure, and subjective bids were compared to each trial's normative information value (derived from task structure) to quantify bidding bias. Two neural networks were trained on this data: a bid-model that predicts human valuations, and a difference-model that predicts deviations from normativity. Clustering the difference-model's penultimate-layer activations revealed 10 overestimation and 17 underestimation bias patterns, which were submitted to a large language model (Claude Sonnet 3.5) for interpretation. The LLM proposed several candidate biases, including the novel undervaluation-of-information-under-complexity bias subsequently validated in Experiments 2 and 3.

### Data Files

- Exp1.csv — Trial-level data: subject ID, trial number, card values (Card1–Card6), normative information value, subjective bid, play/pass choice, BDM draw, trial expected value and variance.

### Main analyses (GitExp1.py)

- Bid distribution conditional on whether information was received vs. not received.
- Distribution of valuation bias (subjective bid − normative information value), with reference lines at zero and at the sample mean.
- Probability of playing the lottery as a function of expected value, fit with a pooled logistic regression and a 95% bootstrap confidence interval.

## Experiment 2

### Background
Experiment 2 tested whether decision complexity predicts the undervaluation of information. Group A (N = 40) classified the average of each six-card array as positive or negative; the mean reaction time across Group A on each array served as an independent, array-level proxy for complexity. Group B (N = 50) then completed the information-seeking task from Experiment 1 on the same 100 arrays. Linear mixed-effects models tested whether array-level complexity predicted trial-by-trial bidding bias, including a covariate model that controlled for expected value and uncertainty (SD of cards). Degrees of freedom were computed using Satterthwaite's method.

### Data Files

- Exp2-Part1.csv — Group A trial-level summaries: TrialNumber, Card1–Card6, Mean_EV, StdDev, Mean_ChoiceRT.
- Exp2-Part2.csv — Group B trial-level summaries: TrialNumber, Card1–Card6, Mean_EV, Std, Mean_InfoSeeking, Mean_InfoValue.
- Exp2.csv — Subject-by-trial data for Group B, used to fit the linear mixed-effects models.

### Main analyses (GitExp2.m)

- Linear mixed-effects model: bidding bias predicted by complexity (RT), with random intercepts and random complexity slopes by subject.
- Covariate linear mixed-effects model: bidding bias predicted by complexity, expected value and uncertainty, with random intercepts and random slopes for all three predictors by subject.

## Experiment 3

### Background
Experiment 3 (N = 50) tested the financial cost of the complexity bias. Eight arrays were selected from Experiment 2: four simple (Group-A RT in the bottom 20%) and four complex (top 20%). Participants made play/pass decisions on each six-card array (no-information condition) and on every three-card subset of each array (information condition), with no bidding stage. Comparing payoffs with vs. without information yielded the empirical value of information for simple and complex arrays separately. The empirical value of information was then compared to the bids participants placed on the same arrays in Experiment 2.

### Data Files

- Exp3.csv — Trial-level data: SubjectID, TrialNumber, Choice, PlayChoice, ReactionTime, Card1–Card6, ExpectedValue, StandardDev, Condition (3 = with information, 6 = no information), HighRT (0 = simple array, 1 = complex array).

### Main analyses (GitExp3.m)

- 2 (complexity: simple vs. complex) × 2 (information: with vs. without) profit analysis under two reward definitions (mean of available cards; single randomly selected card).
- Information valuation bias plot comparing participants' bids in Experiment 2 to the empirical value of information measured in Experiment 3.

## Dependencies

To run the code, you will need:

- Python 3.9 or higher (for GitExp1.py): numpy, pandas, matplotlib, statsmodels, scipy.
- MATLAB R2021a or higher (for GitExp2.m and GitExp3.m), Statistics and Machine Learning Toolbox.

## Contact Information

For any questions or inquiries, please contact:

- Moshe Glickman: mosheglickman345@gmail.com
- Tali Sharot: t.sharot@ucl.ac.uk
