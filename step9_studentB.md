Two key objectives in Bayesian networks are parameter and structure learning. Parameter learning involves determining numerical values that describe the connections between variables in a specific network structure. On the other hand, structure learning is concerned with discovering the network's structure or determining which factors are closely linked.

Parameter learning refers to calculating conditional probability distributions (CPDs) of variables within a Bayesian network. With a network structure, parameter learning aims at determining the probabilities that influence the relations of parent and child nodes.

Assuming a Bayesian network structure that has two nodes: $A$ (parent node) and $B$ (child node). Conditional probability $P(B|A)$ defines the link between $A$ and $B$. If $A$ has two possible states $A_1$ and $A_2$, and $B$ likewise has two possible states $B_1$ and $B_2$, parameter learning requires estimating the probabilities:

- $P(B_1 | A_1)$
- $P(B_2 | A_1)$
- $P(B_1 | A_2)$
- $P(B_2 | A_2)$

These probabilities may be calculated using techniques like maximum likelihood estimation (MLE) and Bayesian estimation.

**Maximum Likelihood Estimation (MLE):**

MLE includes determining the value of parameters which maximize the likelihood for the data. For example, if observed data displays how frequently $A$ and $$ take on particular states, those findings can be used to compute the probabilities which maximize the probability of the data considering the network structure.

**Bayesian Estimation:**

Bayesian estimate uses prior understanding of the parameters and then updates it depending on the observed data. This method includes historical distributions with the probability of data to generate posterior distributions for all the parameters.

Example
Assume a network in which a node $MarketTrend$ affects nodes $StockPrice$ & $TradingVolume$. The conditional probabilities $P(StockPrice|MarketTrend)$ and $P(TradingVolume|MarketTrend)$ have to be determined. Using past financial data, parameter learning approaches like as MLE may estimate these probabilities while assessing the likelihood of various stock prices and trade volumes based on market trends.

### Structured Learning

Structure learning attempts to discover the network structure itself and precisely what variables are closely connected by edges. The technique is considerably complicated than parameter learning since it requires searching through a wide array of network structures to discover the one which reflects the relationships in data most accurately.

**Techniques for Structure Learning:**

1. Score-Based: These approaches use a scoring system to assess network structures, balancing model fit with complexity. The Bayesian Information Criterion (BIC) along with the Akaike Information Criterion (AIC) are two often used scoring functions. The aim is to identify the structure that results in maximizing the score.

2. Constraint-Based:
Constraint-based approaches make use of conditional independence tests to evaluate network structure. These tests determine if 2 variables have been conditionally independent based on a set of other factors. The network is built by connecting variables that are determined as conditionally dependent.

**Differences**

- **Parameter Learning:** Estimates numerical CPD values for a particular network structure.
- **Structure Learning:** Determines the network's structure by recognizing which factors are closely linked.
<br></br>
- **Parameter Learning:** Less computationally demanding since it employs a defined structure to estimate probability.
- **Structure Learning:** Computationally demanding, requiring a search across a space of potential network structures that expands exponentially with the number of variables.
<br></br>
- **Parameter Learning:** Needs a defined network structure for determining parameters.
- **Structured Learning** Can be conducted with no prior knowledge of network structure, however, domain knowledge is typically useful for limiting the search space.

Parameter learning seeks to estimate the CPDs from a network structure, whereas structure learning seeks to reveal the network structure. Both processes are required for the creation of accurate and comprehensible models, with parameter learning giving numeric correlations and structure learning establishing the network's structure. Understanding and using these concepts enables experts to use Bayesian networks to simulate complicated connections.