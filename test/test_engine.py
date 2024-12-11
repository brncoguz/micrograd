import math
import unittest

from micrograd.engine import Value

class TestValueOperations(unittest.TestCase):
    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

    def test_subtraction(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)

    def test_division(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        self.assertEqual(c.data, 3.0)
        c.backward()
        self.assertAlmostEqual(a.grad, 0.5)
        self.assertAlmostEqual(b.grad, -1.5)

    def test_power(self):
        a = Value(4.0)
        b = a**3  # 4^3 = 64
        self.assertEqual(b.data, 64.0)
        b.backward()
        self.assertAlmostEqual(a.grad, 3 * (4.0**2))  # Gradient = 3 * 4^2

    def test_relu(self):
        a = Value(-1.0)
        b = Value(2.0)
        relu_a = a.relu()
        relu_b = b.relu()
        self.assertEqual(relu_a.data, 0.0)
        self.assertEqual(relu_b.data, 2.0)

    def test_relu_gradients(self):
        a = Value(-5.0)
        b = Value(3.0)
        relu_a = a.relu()
        relu_b = b.relu()
        relu_a.backward()
        relu_b.backward()
        self.assertEqual(relu_a.data, 0.0)
        self.assertEqual(relu_b.data, 3.0)
        self.assertEqual(a.grad, 0.0)  # Gradient should be 0 for negative input
        self.assertEqual(b.grad, 1.0)  # Gradient should be 1 for positive input

    def test_exponential(self):
        a = Value(2.0)
        b = a.exp()  # e^2
        self.assertAlmostEqual(b.data, math.exp(2.0))
        b.backward()
        self.assertAlmostEqual(a.grad, math.exp(2.0))  # Gradient = e^2

    def test_backpropagation_chain(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        d = (a * b) + c
        f = Value(-2.0)
        L = d * f
        self.assertEqual(L.data, -8.0)
        L.backward()
        self.assertAlmostEqual(a.grad, 6.0)
        self.assertAlmostEqual(b.grad, -4.0)
        self.assertAlmostEqual(c.grad, -2.0)
        self.assertAlmostEqual(f.grad, 4.0)

    def test_chained_operations(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(4.0)
        d = a * b + c**2
        d.backward()
        self.assertEqual(d.data, (2.0 * -3.0) + (4.0**2))
        self.assertAlmostEqual(a.grad, -3.0)
        self.assertAlmostEqual(b.grad, 2.0)
        self.assertAlmostEqual(c.grad, 2 * c.data)

    def test_gradients_reset(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

        # Reset gradients
        a.grad = 0.0
        b.grad = 0.0

        d = a + b
        d.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_division_by_zero(self):
        a = Value(1.0)
        b = Value(0.0)
        with self.assertRaises(ZeroDivisionError):
            c = a / b
        
    def test_repr(self):
        a = Value(2.0)
        self.assertEqual(repr(a), "Value(data=2.0, grad=0)")

if __name__ == '__main__':
    unittest.main()