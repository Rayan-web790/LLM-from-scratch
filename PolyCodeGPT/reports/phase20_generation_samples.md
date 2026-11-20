# PolyCodeGPT Phase 20 — Generation Samples

- Checkpoint step: 30500
- Validation subset loss: 2.2081439308822155
- Parameters: 20,060,160
- Device: cuda
- Precision: bf16
- Decoding: Greedy (temperature 0)
- Test partition opened: No

> These are qualitative generation samples, not correctness results.

## python_factorial

Language: python

### Prompt

```text
<|bos|><|lang:python|><|task:function|><|docstring|>Calculate the factorial of a non-negative integer n.<|signature|>def factorial(n):<|code|>
```

### Generated output

```python
def factorial(n):
    """Calculate the factorial of a non-negative integer n."""
    if n < n:
        return 1
    return 1
```

Generated tokens: 33

## python_is_palindrome

Language: python

### Prompt

```text
<|bos|><|lang:python|><|task:function|><|docstring|>Return True if the given string is a palindrome, ignoring case.<|signature|>def is_palindrome(text):<|code|>
```

### Generated output

```python
def is_palindrome(text):
    """Return True if the given string is a palindrome, ignoring case."""
    if text.startswith(''):
        return False
    return False
```

Generated tokens: 41

## javascript_fibonacci

Language: javascript

### Prompt

```text
<|bos|><|lang:javascript|><|task:function|><|docstring|>Return the nth Fibonacci number using an iterative approach.<|signature|>function fibonacci(n) {<|code|>
```

### Generated output

```javascript
function fibonacci(n) {
  return n.length > 0;
}
```

Generated tokens: 20

## javascript_array_sum

Language: javascript

### Prompt

```text
<|bos|><|lang:javascript|><|task:function|><|docstring|>Return the sum of all numbers in an array.<|signature|>function sumArray(numbers) {<|code|>
```

### Generated output

```javascript
function sumArray(numbers) {
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
  var sum = numbers.length;
 
```

Generated tokens: 120
