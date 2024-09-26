# Unit Testing Guidelines for Python Projects

## 1. Test Structure and Organization

- Use the `unittest` framework for writing and organizing tests.
- Create separate test classes for different components or functionalities.
- Name test classes descriptively, e.g., `TestTaskDescriptionGeneratorBasic`, `TestTaskDescriptionGeneratorExamples`.
- Use meaningful names for test methods, starting with `test_`, e.g., `test_generate_description`, `test_analyze_input`.

```python
import unittest

class TestComponentName(unittest.TestCase):
    def test_specific_functionality(self):
        # Test code here
```

## 2. Test Setup and Teardown

- Use `setUp` method to initialize objects and set up the test environment.
- Use `tearDown` method if you need to clean up after tests (not shown in the example, but useful for database connections, file handling, etc.).

```python
def setUp(self):
    self.model = ChatOpenAI(model="llama3-70b-8192", temperature=1.0, max_retries=3)
    self.generator = TaskDescriptionGenerator(self.model)
```

## 3. Mocking External Dependencies

- Use `unittest.mock.patch` to mock external dependencies, especially API calls or complex objects.
- Mock at the method level when possible for more granular control.

```python
from unittest.mock import patch

@patch.object(ChatOpenAI, "invoke")
def test_method_with_api_call(self, mock_invoke):
    mock_invoke.return_value = "Mocked response"
    # Test code using the mocked method
```

## 4. Testing Different Scenarios

- Test both valid and invalid inputs.
- Include edge cases and boundary conditions.
- Test error handling and exception raising.

```python
def test_valid_input(self):
    # Test with valid input

def test_invalid_input(self):
    # Test with invalid input

def test_edge_case(self):
    # Test edge case scenario
```

## 5. Assertions

- Use appropriate assertion methods provided by `unittest.TestCase`.
- Common assertions: `assertEqual`, `assertIn`, `assertTrue`, `assertRaises`.

```python
self.assertEqual(result, expected_value)
self.assertIn(item, collection)
self.assertTrue(condition)
with self.assertRaises(ExpectedException):
    # Code that should raise an exception
```

## 6. Parameterized Tests

- For testing multiple similar cases, consider using parameterized tests.
- You can use third-party libraries like `parameterized` or create custom parameterized test methods.

```python
@parameterized.expand([
    ("input1", "expected1"),
    ("input2", "expected2"),
])
def test_parameterized_method(self, input, expected):
    result = self.method_under_test(input)
    self.assertEqual(result, expected)
```

## 7. Testing Asynchronous Code

- For async functions, use `asyncio` and the `unittest.IsolatedAsyncioTestCase`.

```python
class TestAsyncMethods(unittest.IsolatedAsyncioTestCase):
    async def test_async_method(self):
        result = await async_method()
        self.assertEqual(result, expected_value)
```

## 8. Coverage and Edge Cases

- Aim for high test coverage, testing all code paths.
- Include tests for error handling and edge cases.
- Use tools like `coverage.py` to measure and improve test coverage.

## 9. Test Independence

- Ensure each test is independent and can run in isolation.
- Avoid dependencies between tests.
- Reset any shared state in the `setUp` method.

## 10. Continuous Integration

- Integrate unit tests into your CI/CD pipeline.
- Run tests automatically on each commit or pull request.

## 11. Maintenance and Refactoring

- Keep tests up to date as the codebase evolves.
- Refactor tests when refactoring the main code to maintain test relevance and accuracy.

## 12. Documentation

- Include docstrings in test methods to explain the purpose and expectations of each test.
- Use clear and descriptive variable names in tests.

```python
def test_specific_functionality(self):
    """
    Test that specific_functionality correctly handles input and produces expected output.
    """
    # Test code here
```

By following these guidelines, you can create comprehensive and maintainable unit tests for your Python projects, ensuring code quality and reliability.