#### __Risk Management__ 

# Group Work Project 1

## Step 2 

### Problem Statement:

Crude oil prices are pivotal indicators in the global economy, prompting extensive efforts from governments and businesses to forecast their future trends. However, predicting these prices remains a challenging endeavor. Traditional methods such as linear regressions and econometrics are frequently used, but alternative approaches, including structural models and computer-guided analytics, are also explored. Despite these efforts, a consensus on the optimal forecasting approach is still lacking (Ross).

Part of the problem in predicting oil prices, therefore, lies in its sensitivity to disruptions in the global demand and supply of oil. A whole host of factors can affect these prices: geopolitical tensions, economic growth patterns, technological advances, and environmental concerns. This complex interplay is such that changes in supply and demand dynamics can be so sudden and unpredictable that oil prices become erratic (Perry). This is because any economic disruption will lead to magnified effects in the current interconnected global economy. Geopolitical events, consumer behavior changes, or a technological breakthrough in the energy sector will instantly and massively reflect the oil price change. A complex web of these different facts predicts oil prices complex, like a jigsaw puzzle.

Bayesian models are known for their superior predictive accuracy compared to conventional time-series analysis. They are commonly applied in forecasting GDP, inflation, consumer prices, and exchange rates. The Bayesian normal multiple regression model with informative priors is particularly relevant here, incorporating insights from the current and anticipated state of the oil market (Lee & Huh). Bayesian networks are ideally suited for long-term oil price forecasting due to their ability to capture complex relationships and uncertainties inherent in economic and market dynamics. Long-term forecasting requires a methodology capable of handling intricate interdependencies among numerous variables and adapting to evolving conditions over extended periods.

Graphical representations, such as Bayesian networks, offer a valuable framework for visualizing underlying probability structures and developing innovative models. These networks elucidate the relationships among variables, effectively handling complex probability problems. An essential attribute of Bayesian networks is their capacity to identify and incorporate relevant features into the decision-making context. This ensures a comprehensive exploration of all pertinent elements when resolving a problem. Bayesian networks provide greater flexibility and scalability compared to alternative network structures and learning methods. Updating a Bayesian network with new data is straightforward, requiring minimal adjustments. Additionally, the graphical representation of Bayesian networks is easy for both humans and computers to understand, unlike other networks such as neural networks, which pose challenges for human comprehension (Ohri).

With the help of Bayesian networks, integrated with far-reaching sets of data on geopolitical developments, economic growth indicators, and technological and environmental issues, among others, a robust model for long-term forecasting of oil prices can be developed. The chances are high that this may capture complex interdependencies and adjust to changing conditions in a way that is more accurate and insightful than available methodologies do.

The specific problem that the thesis addresses is the need for more accurate and reliable long-term forecasting of crude oil prices. Bayesian networks can be applied in modeling these relationships between different factors that lead to the cost of oil, as it is complex and uncertain. Traditional methods simply cannot handle the dynamic and interdependent nature of these factors; hence, Bayesian networks are the ideal choice for long-term forecasting.

Among the advantages of Bayesian networks is that they make it possible to include prior knowledge and update predictions with new data as it becomes available. This adaptability is an essential point in the forecasting of oil prices because, within months, new geopolitical events or technological improvements may easily change the state of market conditions. In addition, since Bayesian networks are graphical, relationships between variables can be visualized, leading to improved human and computational analysis.

This work aims to create a complete framework for the multiple complexities that arise when trying to forecast the price of crude oil and, therefore, to offer something practical in decision-making within the world economy.


## Key Concept 1: Markov Property 
\begin{align}

P\left[\underbracket{X_{t+1}=j}_{\text{Future state}} \mid \underbracket{X_t=i}_{\text{Present state}}, \underbracket{X_{t-1}=k, \cdots, X_0=m}_{\text{Past states}}\right] \\

= P\left[\underbracket{X_{t+1}=j}_{\text{Future state}} \mid \underbracket{X_t=i}_{\text{Present state}}\right] 



= \underbracket{p_{i j}}_{\text{Transition probability}}
\end{align}

1. This property implies that once the present state is known, the past states do not provide any additional information about the future state.
2. This property implies that once the present state is known, the past states do not provide any additional information about the future state.
3. The behavior of a Markov process can be fully described by specifying the transition probabilities $P(X_{t+1} = j \mid X_t = i)$ for all pairs of states $i, j \in S$. The set $S$ contains all possible states the process can occupy. This set is countable, meaning it can be finite or countably infinite.

> Any stochastic process that follows the Markov property is known as Markov Chain.


## Key Concept 2: Hidden Markov Model
### Key Ideas

###### 1. Hidden State  
$$   Q = \{q_1, \cdots, q_N\}   $$This is a finite set of $N$ states. These states are not directly observable and are considered "hidden." For example: The market can be in different states, such as a bull market (rising prices) or a bear market (falling prices). These regimes are inherently unobservable. 
###### 2. Observable Outputs
These hidden states generate observable outputs or symptom such as falling or rising spot prices:   $$   \Sigma = \{s_1, \cdots, s_M\}   $$
###### 3. Initial Probability Vector
Hidden states are associated are associated with an initial probability vector: 
$$    \Pi = \{\pi_i\} \qquad \text{such that} \qquad \sum_{i=1}^{N} \pi_i = 1   $$
This represents the initial probability of the market being in a bull or bear market. For instance, based on historical data, there may be a higher probability of starting in a bull market.
###### 4. State transition probability matrix
$$   A = \{a_{ij}\}   $$
This is the state transition probability matrix. The element $a_{ij}$ represents the probability of transitioning from state $q_i$ to state $q_j. The probabilities for transitions from any given state must sum to 1:   $$   \sum_{j=1}^{N} a_{ij} = 1 \quad \forall i \in Q   $$This represents the probability of transitioning from one market regime to another. For example, the probability of transitioning from a bull market to a bear market or staying in the same market regime.
###### 5. Emission Probability Matrix   $$   B = \{b_i(v_k)\}   $$
This is the emission probability matrix. The element $b_i(v_k)$ represents the probability of observing symbol $v_k$ when in state $q_i$. The probabilities for observing all possible symbols from any given state must sum to 1:   $$  \sum_{k=1}^{M} b_i(v_k) = 1 \quad \forall i \in Q   $$This represents the probability of observing specific market returns and financial indicators given a particular market regime. For example, the probability of observing positive returns and high trading volume in a bull market.

> **Combining these key ideas, the Hidden Markov Model can be specified as a 5-tuple $(Q, \Sigma, \Pi, A, B)$** 


## Key Concept 3: Directed Acyclic Graph

**What is directed?**
Edges have a direction (from one vertex to another).
```
A → B → C → D → E
```

**What are cycle?**
A cycle refers to a path in the graph that starts and ends at the same vertex, following the direction of the edges.
```
   A <---
   ↓    |
   B → C
   ↓    |
   D ---'

```

Therefore, DAG is directed and has no cycles.


## Key Concept 4: Representation of HMM as DAG

**Nodes:**
1. **Hidden State Nodes**: Represent the hidden states $q_1, q_2, \ldots, q_N$.
2. **Observable Symbol Nodes**: Represent the observable symbols $s_1, s_2, \ldots, s_M$
**Edges**:
  - **Transition Edges**: Directed edges between hidden state nodes $q_i$ and $q_j$ with weights $a_{ij}$ representing transition probabilities.
  - **Emission Edges**: Directed edges from hidden state nodes $q_i$ to observable symbol nodes $s_k$ with weights $b_i(v_k)$ representing emission probabilities.

With two hidden states (q1 and q2) and three observable symbols (s1, s2, s3), the DAG might look like this:

- **Hidden States**:
  -$q1$ and $q2$ are represented as nodes.
  -$q1$ has an edge to $q2$ with weight $a_{12}$.
- **Observable Symbols**:
  - $s1, s2,$ and $s3$ are represented as nodes.
  - $q1$ emits $s1, s2,$ and $s3$ with probabilities $b_1(s1), b_1(s2), b_1(s3)$.
  - $q2$ emits $s1, s2,$ and $s3$ with probabilities $b_2(s1), b_2(s2), b_2(s3)$.

## Key Concept 5: Markov Blanket

For a node $v$, its Markov Blanket $MB(v)$ consists of:
  1. **Parents of $v$**: Nodes directly connected to $v$ with incoming edges.
  2. **Children of $v$**: Nodes directly connected to $v$ with outgoing edges.
  3. **Parents of children of $v$**: Nodes that are parents of any child node of $v$.

If $v$ represents a patient's health condition, its Markov Blanket might include:
  - Tests (nodes that cause $v$),
  - Symptoms (children of $v$),
  - Other conditions that can cause the same health condition (parents of children of $v$).

###### Conditional Independence
