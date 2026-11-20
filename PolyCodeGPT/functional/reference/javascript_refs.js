// Phase 11 trusted reference implementations for the 50 JavaScript tasks.
// Written from functional/manifest.json only; used as the oracle to validate the
// test cases inside the sandbox, never as a source for candidate code.

// ── 01-10 ────────────────────────────────────────────────────────────────────

function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

function fibonacci(n) {
  if (n < 2) return n;
  let previous = 0;
  let current = 1;
  for (let i = 0; i < n - 1; i++) {
    const next = previous + current;
    previous = current;
    current = next;
  }
  return current;
}

function isPalindrome(s) {
  let reversed = "";
  for (let i = s.length - 1; i >= 0; i--) reversed += s[i];
  return s === reversed;
}

function mergeSorted(a, b) {
  const merged = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] <= b[j]) merged.push(a[i++]);
    else merged.push(b[j++]);
  }
  while (i < a.length) merged.push(a[i++]);
  while (j < b.length) merged.push(b[j++]);
  return merged;
}

function binarySearch(arr, target) {
  let low = 0;
  let high = arr.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) low = mid + 1;
    else high = mid - 1;
  }
  return -1;
}

function nthPrime(n) {
  let count = 0;
  let candidate = 1;
  while (count < n) {
    candidate++;
    if (isPrime(candidate)) count++;
  }
  return candidate;
}

function reverseString(s) {
  let out = "";
  for (let i = s.length - 1; i >= 0; i--) out += s[i];
  return out;
}

function isAnagram(s1, s2) {
  if (s1.length !== s2.length) return false;
  const counts = {};
  for (const ch of s1) counts[ch] = (counts[ch] || 0) + 1;
  for (const ch of s2) {
    if (!counts[ch]) return false;
    counts[ch] -= 1;
  }
  return true;
}

function gcd(a, b) {
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function formatDate(dateStr) {
  const parts = dateStr.split("-");
  return `${parts[2]}/${parts[1]}/${parts[0]}`;
}

// ── 11-20 ────────────────────────────────────────────────────────────────────

function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false;
  }
  return true;
}

function triangularNumber(n) {
  return (n * (n + 1)) / 2;
}

function linearSearch(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
  }
  return -1;
}

function findMax(arr) {
  let best = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > best) best = arr[i];
  }
  return best;
}

function bubbleSort(arr) {
  const result = arr.slice();
  for (let i = 0; i < result.length; i++) {
    let swapped = false;
    for (let j = 0; j < result.length - 1 - i; j++) {
      if (result[j] > result[j + 1]) {
        const t = result[j];
        result[j] = result[j + 1];
        result[j + 1] = t;
        swapped = true;
      }
    }
    if (!swapped) break;
  }
  return result;
}

function isEven(n) {
  return n % 2 === 0;
}

function sumArray(arr) {
  let total = 0;
  for (const value of arr) total += value;
  return total;
}

function factorialIter(n) {
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

function myLen(s) {
  let count = 0;
  while (s.charAt(count) !== "") count++;
  return count;
}

function isLeapYear(y) {
  if (y % 400 === 0) return true;
  if (y % 100 === 0) return false;
  return y % 4 === 0;
}

// ── 21-30 ────────────────────────────────────────────────────────────────────

function countdown(n) {
  const out = [];
  for (let i = n; i >= 1; i--) out.push(i);
  return out;
}

function safeFactorial(n) {
  if (n < 0) throw new RangeError("factorial is undefined for negative integers");
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

function shiftChar(ch, n) {
  const code = ch.charCodeAt(0);
  if (code >= 97 && code <= 122) return String.fromCharCode(((code - 97 + n) % 26 + 26) % 26 + 97);
  if (code >= 65 && code <= 90) return String.fromCharCode(((code - 65 + n) % 26 + 26) % 26 + 65);
  return ch;
}

function caesarCipher(s, n) {
  let out = "";
  for (const ch of s) out += shiftChar(ch, n);
  return out;
}

function fibMemo(n, memo) {
  if (memo === undefined) memo = {};
  if (memo[n] !== undefined) return memo[n];
  if (n < 2) {
    memo[n] = n;
    return n;
  }
  memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
  return memo[n];
}

function getEvens(arr) {
  return arr.filter((value) => value % 2 === 0);
}

function getOdds(arr) {
  return arr.filter((value) => Math.abs(value % 2) === 1);
}

function filterPositives(arr) {
  return arr.filter((value) => value > 0);
}

function stringLengths(arr) {
  return arr.map((value) => myLen(value));
}

function isSorted(arr) {
  for (let i = 1; i < arr.length; i++) {
    if (arr[i - 1] > arr[i]) return false;
  }
  return true;
}

function fibRecursive(n) {
  if (n < 2) return n;
  return fibRecursive(n - 1) + fibRecursive(n - 2);
}

// ── 31-40 ────────────────────────────────────────────────────────────────────

function removeDuplicates(arr) {
  const out = [];
  for (const value of arr) {
    if (!out.includes(value)) out.push(value);
  }
  return out;
}

function listIntersection(arr1, arr2) {
  const out = [];
  for (const value of arr1) {
    if (arr2.includes(value) && !out.includes(value)) out.push(value);
  }
  return out;
}

function listUnion(arr1, arr2) {
  const out = [];
  for (const value of arr1.concat(arr2)) {
    if (!out.includes(value)) out.push(value);
  }
  return out;
}

function flatten(arr) {
  const out = [];
  for (const item of arr) {
    if (Array.isArray(item)) {
      for (const nested of flatten(item)) out.push(nested);
    } else {
      out.push(item);
    }
  }
  return out;
}

function includesSorted(arr, target) {
  return binarySearch(arr, target) !== -1;
}

function runningTotal(arr) {
  let total = 0;
  const out = [];
  for (const value of arr) {
    total += value;
    out.push(total);
  }
  return out;
}

function selectionSort(arr) {
  const result = arr.slice();
  for (let i = 0; i < result.length; i++) {
    let smallest = i;
    for (let j = i + 1; j < result.length; j++) {
      if (result[j] < result[smallest]) smallest = j;
    }
    const t = result[i];
    result[i] = result[smallest];
    result[smallest] = t;
  }
  return result;
}

function caesarWrap(s, n) {
  let out = "";
  for (const ch of s) out += shiftChar(ch, n);
  return out;
}

function sameWord(w1, w2) {
  return w1.toLowerCase() === w2.toLowerCase();
}

function myAbs(n) {
  return n < 0 ? -n : n;
}

// ── 41-50 ────────────────────────────────────────────────────────────────────

function power(base, exp) {
  let result = 1;
  for (let i = 0; i < exp; i++) result *= base;
  return result;
}

function isDigits(s) {
  if (s.length === 0) return false;
  for (const ch of s) {
    if (ch < "0" || ch > "9") return false;
  }
  return true;
}

function minOfThree(a, b, c) {
  let smallest = a;
  if (b < smallest) smallest = b;
  if (c < smallest) smallest = c;
  return smallest;
}

function cToF(c) {
  return (c * 9) / 5 + 32;
}

function isPositive(n) {
  return n > 0;
}

function nthEven(n) {
  return 2 * n;
}

function myMod(a, b) {
  return a - Math.floor(a / b) * b;
}

function isBlank(s) {
  return s.trim() === "";
}

function square(n) {
  return n * n;
}

function fibIter(n) {
  let previous = 0;
  let current = 1;
  for (let i = 0; i < n; i++) {
    const next = previous + current;
    previous = current;
    current = next;
  }
  return previous;
}
