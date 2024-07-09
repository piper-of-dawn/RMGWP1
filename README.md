#### __Risk Management__ 

# Group Work Project 1

## Problem and Overview of Bayesian networks

### Problem Statement:

Crude oil prices are pivotal indicators in the global economy, prompting extensive efforts from governments and businesses to forecast their future trends. However, predicting these prices remains a challenging endeavor. Traditional methods such as linear regressions and econometrics are frequently used, but alternative approaches, including structural models and computer-guided analytics, are also explored. Despite these efforts, a consensus on the optimal forecasting approach is still lacking (Ross).

Part of the problem in predicting oil prices, therefore, lies in its sensitivity to disruptions in the global demand and supply of oil. A whole host of factors can affect these prices: geopolitical tensions, economic growth patterns, technological advances, and environmental concerns. This complex interplay is such that changes in supply and demand dynamics can be so sudden and unpredictable that oil prices become erratic (Perry). This is because any economic disruption will lead to magnified effects in the current interconnected global economy. Geopolitical events, consumer behavior changes, or a technological breakthrough in the energy sector will instantly and massively reflect the oil price change. A complex web of these different facts predicts oil prices complex, like a jigsaw puzzle.

Bayesian models are known for their superior predictive accuracy compared to conventional time-series analysis. They are commonly applied in forecasting GDP, inflation, consumer prices, and exchange rates. The Bayesian normal multiple regression model with informative priors is particularly relevant here, incorporating insights from the current and anticipated state of the oil market (Lee & Huh). Bayesian networks are ideally suited for long-term oil price forecasting due to their ability to capture complex relationships and uncertainties inherent in economic and market dynamics. Long-term forecasting requires a methodology capable of handling intricate interdependencies among numerous variables and adapting to evolving conditions over extended periods.

Graphical representations, such as Bayesian networks, offer a valuable framework for visualizing underlying probability structures and developing innovative models. These networks elucidate the relationships among variables, effectively handling complex probability problems. An essential attribute of Bayesian networks is their capacity to identify and incorporate relevant features into the decision-making context. This ensures a comprehensive exploration of all pertinent elements when resolving a problem. Bayesian networks provide greater flexibility and scalability compared to alternative network structures and learning methods. Updating a Bayesian network with new data is straightforward, requiring minimal adjustments. Additionally, the graphical representation of Bayesian networks is easy for both humans and computers to understand, unlike other networks such as neural networks, which pose challenges for human comprehension (Ohri).

With the help of Bayesian networks, integrated with far-reaching sets of data on geopolitical developments, economic growth indicators, and technological and environmental issues, among others, a robust model for long-term forecasting of oil prices can be developed. The chances are high that this may capture complex interdependencies and adjust to changing conditions in a way that is more accurate and insightful than available methodologies do.

The specific problem that the thesis addresses is the need for more accurate and reliable long-term forecasting of crude oil prices. Bayesian networks can be applied in modeling these relationships between different factors that lead to the cost of oil, as it is complex and uncertain. Traditional methods simply cannot handle the dynamic and interdependent nature of these factors; hence, Bayesian networks are the ideal choice for long-term forecasting.

Among the advantages of Bayesian networks is that they make it possible to include prior knowledge and update predictions with new data as it becomes available. This adaptability is an essential point in the forecasting of oil prices because, within months, new geopolitical events or technological improvements may easily change the state of market conditions. In addition, since Bayesian networks are graphical, relationships between variables can be visualized, leading to improved human and computational analysis.

This work aims to create a complete framework for the multiple complexities that arise when trying to forecast the price of crude oil and, therefore, to offer something practical in decision-making within the world economy.


## Suitability of bayesian networks for oil price forecasting

1. Dealing with uncertain and complex relationships: Estimating the price of crude oil requires considering a wide range of interrelated macroeconomic, geopolitical, and market variables. When modeling complex structures with several variables and uncertain interactions, Bayesian networks perform exceptionally well. According to the thesis, the complex relationship of global economic factors determines the price of crude oil. This covers the quantity of oil produced by nations like those in OPEC, the quantities of oil used by the developed economies (OECD), and the continuous political and economic developments that take place globally. Bayesian networks are perfect for depicting the complexities of the oil market because they can capture these complicated interactions and interdependence amongst the variables within a graphical structure.

2. Understanding the market structure: One of the primary issues addressed in the thesis is understanding the workings of the oil markets. Bayesian networks are capable of learning a model's structure and parameters from its data. The paper explores two primary methodologies for structure learning using Bayesian networks:

   a) Constraint-based learning: This method uses statistical tests to discover conditional independencies amongst variables and then builds a network that fulfills the constraints.

   b) Score-based learning: This approach examines a set of alternative graph topologies to discover the one that best matches the data based on a scoring system. 

3. Addressing inaccurate, missing, or incomplete economic and financial information often results in noise. Given that Bayesian networks can execute probabilistic inference with imperfect knowledge, they are resilient to noisy or incomplete data. Due to this, they are especially well-suited for practical uses in financial markets, where data quality might fluctuate.

4. Including domain information: Domain expertise can be incorporated into the model using Bayesian networks. This means that domain knowledge from market specialists, energy strategists, and economists can be added to the prior probability or network structure in light of oil price predictions. Expert insight and data-driven learning together have the potential to produce models that are more precise and understandable.

5. Probabilistic estimations: Bayesian networks, as opposed to deterministic models, produce probabilistic estimates, which are more valuable for making decisions in unpredictable contexts like financial markets. The goal of the thesis is to present an accurate crude oil price estimate, and belief networks can offer probability distributions and confidence intervals for future oil prices in addition to point estimates.
6. Causality: Analyzing the nuances of the oil market requires the ability to depict causal linkages between variables, something Bayesian networks can provide. The aim of establishing a probabilistic graphical model to illustrate the motion of the oil market and establish the causal link between these many variables is addressed in the paper. In the energy industry, this causal reasoning capacity can help with making policies and strategic choices by enabling better interpretation of models.

7. Resilience towards fresh data: When new data is accessible, Bayesian networks may be adjusted effectively. This is especially crucial in the oil industry, which is changing swiftly and is susceptible to price changes due to market trends, geopolitical events, and new economic data. Bayesian networks are well-suited for keeping forecasts current because of their capacity to integrate new information and change probability.

8. Managing numerous time scales: A variety of time-scale elements, ranging from short-term supply interruptions to long-term changes in the economy, have an impact on the oil market. Bayesian networks can describe temporal relationships and produce multi-quarter predictions, especially when paired with methods like the Dynamic Bayesian Networks or the Hidden Markov Models.

9. Research and application: A further issue described in the paper is the research and exploitation of existing data and learned structures for forecasting oil market behavior. Belief networks are appropriate for this purpose because they can effectively execute inference on the structure, enabling both variable exploration and prediction.

10. Model credibility: The paper goes on to discuss ways to validate the developed model's performance and dependability. Bayesian networks offer a variety of model validation techniques, like cross-validation, sensitivity analyses, and posterior predictive checks. These strategies can aid in determining the of the model and credibility before implementation in financial markets.

11. In contrast to certain black box neural network algorithms, Bayesian networks show the correlations between variables graphically. This interpretability is critical for financial applications, whereby understanding the logic behind forecasts is sometimes as essential as the estimates themselves. The paper seeks to gain insight into the operations of the oil markets and the belief networks aid this purpose by offering perspectives on the market's structure and behavior.

12. Nonlinear connections amongst the variables are common in oil pricing. Bayesian networks may capture these nonlinear relationships using conditional probability tables and continuous distributions, resulting in a more accurate picture of the oil market's complex relations.

13. With the Bayesian networks, variables may be easily manipulated to perform hypothetical studies and stress tests. This is consistent with the paper's purpose of replicating economic hardship scenarios to test the model's dependability. Analysts can investigate how alternative scenarios may affect oil prices and evaluate how the model performs under various situations by modifying the values or probability of specific variables. The paper addresses using data from a variety of sources, such as the Energy Information Administration (EIA) and the Federal Reserve Economic Data (FRED). Bayesian networks may successfully integrate information from numerous sources, merging disparate datasets to create one unified model of the oil market. Due to the many factors impacting the price of oil, the issue naturally requires data that is highly dimensional. Bayesian networks can effectively produce and infer highly dimensional probability distributions, which makes them ideal for this challenging forecasting challenge.

## Data

### Extreme Outlier Treatment
[STUDENT A TO WRITE]

### Bad Data Treatment
[STUDENT A TO WRITE]

### Treatment of Missing Values

Some datasets primarily the political stability data was collected on a yearly basis. Therefore, to address this limitation, we adopted the linear interpolation methodology to transform the dataset from a yearly resolution to a daily resolution. 

Mathematically, linear interpolation can be expressed as follows:

$$y = y_1 + \frac{(x - x_1)(y_2 - y_1)}{(x_2 - x_1)}$$

Where:
-$(x_1, y_1)$ and $(x_2, y_2)$ are the known data points,
-$x$ is the point at which the value is to be estimated,
-$y$ is the interpolated value at $x$.

In the context of our study,$x$ represents the date for which the value is being interpolated, and $y$ represents the corresponding data value.

The interpolation process ensures a smooth transition between data points, effectively approximating the daily values while maintaining the overall trend and seasonality inherent in the original yearly data. The result is a finely granulated dataset suitable for detailed analysis and modeling.

The following code achieved this:

```python
def interpolate_data (df, method:str):
    return pl.DataFrame(
        {
            "period": pl.date_range(df["period"].min(), df["period"].max(), '1d', eager=True).alias('period'),
        }
    ).join(df, on="period", how="left").with_columns(pl.exclude('period').interpolate("linear"))
```

## Data Visualisation

### Time Series Plots

The time series plots are shown as follows:

![](./time-series_plots/fuels-consumption-over-time.png)
![](./time-series_plots/interest-rate-over-time.png)
![](./time-series_plots/net_inventory_withdrawals-over-time.png)
![](./time-series_plots/production-capacity-over-time.png)
![](./time-series_plots/total-petroleum-supply-over-time.png)
![](./time-series_plots/unemployment-over-time.png)


### Distributional and Multivariate Kernel Density Plots
In our study, we subsetted the highly correlated variables to enhance the clarity and interpretability of the multivariate density plots. The plots are presented as follows:


![](./multivariate_plots/multivariate_density_0.png)
![](./multivariate_plots/multivariate_density_1.png)
![](./multivariate_plots/multivariate_density_10.png)
![](./multivariate_plots/multivariate_density_11.png)
![](./multivariate_plots/multivariate_density_12.png)
![](./multivariate_plots/multivariate_density_13.png)
![](./multivariate_plots/multivariate_density_14.png)
![](./multivariate_plots/multivariate_density_15.png)
![](./multivariate_plots/multivariate_density_16.png)
![](./multivariate_plots/multivariate_density_17.png)
![](./multivariate_plots/multivariate_density_18.png)
![](./multivariate_plots/multivariate_density_19.png)
![](./multivariate_plots/multivariate_density_2.png)
![](./multivariate_plots/multivariate_density_20.png)
![](./multivariate_plots/multivariate_density_21.png)
![](./multivariate_plots/multivariate_density_22.png)
![](./multivariate_plots/multivariate_density_23.png)
![](./multivariate_plots/multivariate_density_24.png)
![](./multivariate_plots/multivariate_density_25.png)
![](./multivariate_plots/multivariate_density_26.png)
![](./multivariate_plots/multivariate_density_27.png)
![](./multivariate_plots/multivariate_density_28.png)
![](./multivariate_plots/multivariate_density_29.png)
![](./multivariate_plots/multivariate_density_3.png)
![](./multivariate_plots/multivariate_density_30.png)
![](./multivariate_plots/multivariate_density_31.png)
![](./multivariate_plots/multivariate_density_32.png)
![](./multivariate_plots/multivariate_density_33.png)
![](./multivariate_plots/multivariate_density_34.png)
![](./multivariate_plots/multivariate_density_35.png)
![](./multivariate_plots/multivariate_density_4.png)
![](./multivariate_plots/multivariate_density_5.png)
![](./multivariate_plots/multivariate_density_6.png)
![](./multivariate_plots/multivariate_density_7.png)
![](./multivariate_plots/multivariate_density_8.png)
![](./multivariate_plots/multivariate_density_9.png)


## Oil Prices
![](./oil_price_plots/oil_price.png)

Oil prices exhibit significant volatility clustering, a phenomenon characterized by periods of high and low volatility. 

![](./oil_price_plots/acf_and_pacf.png)
The PACF of oil prices reveals that significant autocorrelations taper off after lag 3, indicating that each oil price return is directly influenced by its immediate three preceding returns (when controlled for preceding lags), with minimal direct effect beyond this horizon.


## Model

### Probabilistic Graphical Models
[STUDENT A TO WRITE]

### Parameter Learning
[STUDENT B TO WRITE]

### Markov Chains and Blankets

##### Key Concept 1: Markov Property

$$
\begin{align}
P\left[\underbrace{X_{t+1}=j}_{\text{Future state}} \mid \underbrace{X_t=i}_{\text{Present state}}, \underbrace{X_{t-1}=k, \cdots, X_0=m}_{\text{Past states}}\right] \\
= P\left[\underbrace{X_{t+1}=j}_{\text{Future state}} \mid \underbrace{X_t=i}_{\text{Present state}}\right] 
= \underbrace{p_{i j}}_{\text{Transition probability}}
\end{align}
$$

1. This property implies that once the present state is known, the past states do not provide any additional information about the future state.
2. This property implies that once the present state is known, the past states do not provide any additional information about the future state.
3. The behavior of a Markov process can be fully described by specifying the transition probabilities $P(X_{t+1} = j \mid X_t = i)$ for all pairs of states $i, j \in S$. The set $S$ contains all possible states the process can occupy. This set is countable, meaning it can be finite or countably infinite.

> Any stochastic process that follows the Markov property is known as Markov Chain.


#### Key Concept 2: Hidden Markov Model

###### 1. Hidden State  

$$ Q = \{q_1, \cdots, q_N\} $$ 
This is a finite set of $N$ states. These states are not directly observable and are considered "hidden." For example: The market can be in different states, such as a bull market (rising prices) or a bear market (falling prices). These regimes are inherently unobservable. 

###### 2. Observable Outputs
These hidden states generate observable outputs or symptom such as falling or rising spot prices:   $\Sigma = \{s_1, \cdots, s_M\}$

###### 3. Initial Probability Vector
Hidden states are associated are associated with an initial probability vector: 
$$\Pi = \{\pi_i\} \qquad \text{such that} \qquad \sum_{i=1}^{N} \pi_i = 1$$
This represents the initial probability of the market being in a bull or bear market. For instance, based on historical data, there may be a higher probability of starting in a bull market.

###### 4. State transition probability matrix
$$   A = \{a_{ij}\}   $$
This is the state transition probability matrix. The element $a_{ij}$ represents the probability of transitioning from state $q_i$ to state $q_j$. The probabilities for transitions from any given state must sum to 1:   $\sum_{j=1}^{N} a_{ij} = 1 \quad \forall i \in Q$. This represents the probability of transitioning from one market regime to another. For example, the probability of transitioning from a bull market to a bear market or staying in the same market regime.

###### 5. Emission Probability Matrix   $B = \{b_i(v_k)\}$
This is the emission probability matrix. The element $b_i(v_k)$ represents the probability of observing symbol $v_k$ when in state $q_i$. The probabilities for observing all possible symbols from any given state must sum to 1:   $\sum_{k=1}^{M} b_i(v_k) = 1 \quad \forall i \in Q$. This represents the probability of observing specific market returns and financial indicators given a particular market regime. For example, the probability of observing positive returns and high trading volume in a bull market.

> **Combining these key ideas, the Hidden Markov Model can be specified as a 5-tuple $(Q, \Sigma, \Pi, A, B)$** 


#### Key Concept 3: Directed Acyclic Graph

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


#### Key Concept 4: Representation of HMM as DAG

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

#### Markov Blanket

For a node $v$, its Markov Blanket $MB(v)$ consists of:
  1. **Parents of $v$**: Nodes directly connected to $v$ with incoming edges.
  2. **Children of $v$**: Nodes directly connected to $v$ with outgoing edges.
  3. **Parents of children of $v$**: Nodes that are parents of any child node of $v$.

If $v$ represents a patient's health condition, its Markov Blanket might include:
  - Tests (nodes that cause $v$),
  - Symptoms (children of $v$),
  - Other conditions that can cause the same health condition (parents of children of $v$).

