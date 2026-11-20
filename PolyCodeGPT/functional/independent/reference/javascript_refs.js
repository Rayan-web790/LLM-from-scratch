function totalOfArray(arr) {
  let total = 0;
  for (const x of arr) total += x;
  return total;
}

function titleCaseWords(s) {
  return s.split(/(\s+)/).map(function (w) {
    if (!w) return w;
    return w.charAt(0).toUpperCase() + w.slice(1);
  }).join("");
}

function distinctValues(arr) {
  const seen = new Set();
  const out = [];
  for (const x of arr) {
    if (!seen.has(x)) { seen.add(x); out.push(x); }
  }
  return out;
}

function flattenOnce(arr) {
  const out = [];
  for (const x of arr) {
    if (Array.isArray(x)) out.push(...x);
    else out.push(x);
  }
  return out;
}

function isDivisibleByTwo(n) {
  return n % 2 === 0;
}

function repeatText(s, n) {
  let out = "";
  for (let i = 0; i < n; i++) out += s;
  return out;
}

function maxOfArray(arr) {
  if (arr.length === 0) return null;
  let m = arr[0];
  for (const x of arr) if (x > m) m = x;
  return m;
}

function countOccurrences(arr, val) {
  let c = 0;
  for (const x of arr) if (x === val) c++;
  return c;
}

function flipString(s) {
  return s.split("").reverse().join("");
}

function computeMean(arr) {
  if (arr.length === 0) return 0;
  let total = 0;
  for (const x of arr) total += x;
  return total / arr.length;
}
