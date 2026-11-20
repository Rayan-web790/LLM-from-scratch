function sumSquaredValues(arr) {
  let t = 0;
  for (const x of arr) t += x * x;
  return t;
}
function countPositive(arr) { return arr.filter(x => x > 0).length; }
function countNegative(arr) { return arr.filter(x => x < 0).length; }
function maxAbsValue(arr) {
  if (arr.length === 0) return null;
  return Math.max(...arr.map(x => Math.abs(x)));
}
function minAbsValue(arr) {
  if (arr.length === 0) return null;
  return Math.min(...arr.map(x => Math.abs(x)));
}
function allPositiveValues(arr) { return arr.every(x => x > 0); }
function anyNegativeValue(arr) { return arr.some(x => x < 0); }
function zeroCount(arr) { return arr.filter(x => x === 0).length; }
function sumFirstN(arr, n) { return arr.slice(0, n).reduce((a, b) => a + b, 0); }
function sumLastN(arr, n) { if (n === 0) return 0; return arr.slice(-n).reduce((a, b) => a + b, 0); }
function roundedMean(arr) {
  if (arr.length === 0) return 0;
  return Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) * 100) / 100;
}
function simpleMedian(arr) {
  if (arr.length === 0) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  if (s.length % 2) return s[m];
  return (s[m - 1] + s[m]) / 2;
}
function valueRange(arr) {
  if (arr.length === 0) return 0;
  return Math.max(...arr) - Math.min(...arr);
}
function minMaxNormalize(arr) {
  if (arr.length === 0) return [];
  const lo = Math.min(...arr), hi = Math.max(...arr);
  if (lo === hi) return arr.map(() => 0);
  return arr.map(x => (x - lo) / (hi - lo));
}
function reverseWordOrder(s) { return s.trim().split(/\s+/).reverse().join(" "); }
function countWordsInText(s) { return s.trim() === "" ? 0 : s.trim().split(/\s+/).length; }
function longestWord(s) {
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length === 0) return "";
  let best = words[0];
  for (const w of words) if (w.length > best.length) best = w;
  return best;
}
function shortestWord(s) {
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length === 0) return "";
  let best = words[0];
  for (const w of words) if (w.length < best.length) best = w;
  return best;
}
function sortedWords(s) { return s.split(/\s+/).filter(Boolean).sort(); }
function stripPunctuation(s) { return s.replace(/[^a-zA-Z0-9 ]/g, ""); }
function isSubstring(a, b) { return b.indexOf(a) !== -1; }
function countSubstring(s, sub) {
  if (sub === "") return 0;
  let count = 0, i = 0;
  while ((i = s.indexOf(sub, i)) !== -1) { count++; i += sub.length; }
  return count;
}
function replaceAllOccurrences(s, oldStr, newStr) {
  return s.split(oldStr).join(newStr);
}
function startsWithVowel(s) { return /^[aeiou]/i.test(s); }
function endsWithDigit(s) { return /[0-9]$/.test(s); }
function alternateCase(s) {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    out += i % 2 === 0 ? s[i].toUpperCase() : s[i].toLowerCase();
  }
  return out;
}
function collapseWhitespaceRuns(s) { return s.trim().replace(/\s+/g, " "); }
function truncateText(s, n) {
  if (s.length <= n) return s;
  return s.slice(0, n) + "...";
}
function padStringLeft(s, width, ch) {
  ch = ch || " ";
  if (s.length >= width) return s;
  return ch.repeat(width - s.length) + s;
}
function isPrimeNumber(n) {
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;
  return true;
}
function nextPrimeNumber(n) {
  if (n < 2) n = 2;
  while (!isPrimeNumber(n)) n++;
  return n;
}
function primeCount(n) {
  let c = 0;
  for (let i = 2; i <= n; i++) if (isPrimeNumber(i)) c++;
  return c;
}
function gcdPair(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { const t = b; b = a % b; a = t; }
  return a;
}
function lcmPair(a, b) {
  if (a === 0 || b === 0) return 0;
  return Math.abs(a * b) / gcdPair(a, b);
}
function iterativeFactorial(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}
function powerOf(base, exp) {
  let r = 1;
  for (let i = 0; i < exp; i++) r *= base;
  return r;
}
function floorSqrt(n) {
  if (n < 0) return 0;
  let i = 0;
  while ((i + 1) * (i + 1) <= n) i++;
  return i;
}
function absoluteDifference(a, b) { return Math.abs(a - b); }
function signOf(n) { return n < 0 ? -1 : (n > 0 ? 1 : 0); }
function clampArray(arr, lo, hi) { return arr.map(x => Math.max(lo, Math.min(hi, x))); }
function positiveFilter(arr) { return arr.filter(x => x > 0); }
function negativeFilter(arr) { return arr.filter(x => x < 0); }
function squareArray(arr) { return arr.map(x => x * x); }
function cubeArray(arr) { return arr.map(x => x * x * x); }
function doubleArray(arr) { return arr.map(x => x * 2); }
function negateArray(arr) { return arr.map(x => -x); }
function ascendingSort(arr) { return arr.slice().sort((a, b) => a - b); }
function descendingSort(arr) { return arr.slice().sort((a, b) => b - a); }
function uniqueSorted(arr) { return Array.from(new Set(arr)).sort((a, b) => a - b); }
function productList(arr) { return arr.reduce((a, b) => a * b, 1); }
