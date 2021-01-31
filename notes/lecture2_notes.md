# Background and Gradient Estimators

what recently became easy in ML?

- training with continuous latent variables to model or generate high-dimensional data

## Why are the easy things easy?

- gradients very informative with lots of parameters
- backprop relatively efficient
- local optima not huge issue in high-dimensional spaces
    - most zero gradient locations are saddle points in high spaces, if you find a local optima, its probably global
    
## Why are the hard things hard?

- much more difficult to handle discrete hiddens or parameters
- cant use backprop through discrete variables
- cant find directions to improve in
- dont use knowledge of the structure of the function being optimized
- basically optimizing black box functions


Any problems that have a large search space, well-defined objective cannot be evaluated on partial inputs.

Examples of hard ML problems:

- GANs to generate text
- VAEs with discrete latents
- Multi-agent communication with words
- Agents that make discrete action choices
- Generative models of structured objects w/ arbitrary size (programs, graphs, large text)

The course covers state of the art in:

- MCTS
- SAT Solving
- Program Induction
- Planning
- Curriculum Learning
- Adaptive search algorithms

## Backprop through discrete variables

Bayesian optimization doesnt scale yet:

- expensive, many model evals
- global surrogates bad in high dimensions
- gradient based optimization is competitive
- can we add cheap model-based opt to REINFORCE?

REINFORCE: $\hat{g}_{REINFORCE}(f) = f(b) \frac{\partial}{\partial \theta} \log p(b | \theta)$

- also called the "score-function estimator", lets us estimate gradients of expectation expressions
- pro: unbiased estimate of gradient
- pro: valid for all f
- con: high variance