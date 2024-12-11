Micrograd Clone

A lightweight Python library for building and training neural networks using a micro-framework for automatic differentiation. Inspired by the original micrograd, this project reimplements core functionalities with some additional enhancements and test cases.

Features
	•	Core Differentiation: Supports basic operations (+, -, *, /, **, relu, exp) with backpropagation.
	•	Computation Graph Visualization: Generate computational graphs using draw_dot.
	•	Unit Testing: Comprehensive test suite to ensure correctness and robustness.
	•	Learning Tool: Great for understanding the inner workings of neural network backpropagation.

Usage

Basic Example

from micrograd.engine import Value

# Define inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Define weights and bias
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.7, label='b')

# Compute the output
x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + b
o = n.relu()

# Backpropagation
o.backward()

# Visualize the computation graph
from micrograd.visualize import draw_dot
draw_dot(o)

Testing

To run the unit tests and ensure the integrity of the system:

python -m unittest discover -s test -p "*.py"