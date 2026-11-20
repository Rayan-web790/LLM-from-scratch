# PolyCodeGPT Phase 20 — Teacher-Forced Validation

- Checkpoint step: 30500
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Precision: BF16
- Seed: 20261119
- Samples per language: 8
- Scoring: completion-only next-token cross-entropy
- Prompt tokens excluded from loss: Yes
- EOS included: Yes
- Test partition opened: No

> Teacher-forced perplexity measures next-token prediction with the correct preceding context. It does not measure free-running generation quality.

## Summary

| Language | Samples | Tokens | Mean loss | Perplexity | Token accuracy |
|---|---:|---:|---:|---:|---:|
| python | 8 | 813 | 1.58938 | 4.90071 | 0.6839 |
| javascript | 8 | 867 | 1.87196 | 6.50103 | 0.6171 |

## Individual results

### python — sample 1

- Record: `2773a538562966619d4dce19402e062f`
- Source: `lmnotify/lmnotify.py`
- Completion tokens: 90
- Mean loss: 1.17941
- Perplexity: 3.25244
- Token accuracy: 0.7444

### python — sample 2

- Record: `9ad77be25aafdce6c56ab55e93c3bbe7`
- Source: `pygeotools/lib/geolib.py`
- Completion tokens: 113
- Mean loss: 2.66011
- Perplexity: 14.29782
- Token accuracy: 0.5487

### python — sample 3

- Record: `704cfb72f41c8fd98167e3343fe370f8`
- Source: `redisco/containers.py`
- Completion tokens: 59
- Mean loss: 1.00023
- Perplexity: 2.71892
- Token accuracy: 0.7627

### python — sample 4

- Record: `07ff8bc39df0ed96cd68d298ebd77f8a`
- Source: `src/redis_lock/__init__.py`
- Completion tokens: 53
- Mean loss: 1.73999
- Perplexity: 5.69727
- Token accuracy: 0.6792

### python — sample 5

- Record: `ebbbcb370eca10ba1003dfac9cac8f83`
- Source: `pyramid_zipkin/request_helper.py`
- Completion tokens: 168
- Mean loss: 1.53825
- Perplexity: 4.65644
- Token accuracy: 0.7083

### python — sample 6

- Record: `d337f5e645e780374110e30ea05ea5a2`
- Source: `fluent_contents/extensions/pluginpool.py`
- Completion tokens: 60
- Mean loss: 0.77986
- Perplexity: 2.18117
- Token accuracy: 0.7833

### python — sample 7

- Record: `94f730daa536ae3771a8ab10f2bd5a1b`
- Source: `bakery/management/commands/build.py`
- Completion tokens: 164
- Mean loss: 2.01321
- Perplexity: 7.48734
- Token accuracy: 0.5976

### python — sample 8

- Record: `099edd4601509535dd780f95bec218a9`
- Source: `setup.py`
- Completion tokens: 106
- Mean loss: 0.93216
- Perplexity: 2.53999
- Token accuracy: 0.7736

### javascript — sample 1

- Record: `fbc34ade7f69ef7a10da22d0241244da`
- Source: `src/queries/label-text.js`
- Completion tokens: 164
- Mean loss: 2.00132
- Perplexity: 7.39879
- Token accuracy: 0.6402

### javascript — sample 2

- Record: `c67f254b6e3ffb998f8af2341660ed37`
- Source: `src/client/gmeServerRequests.js`
- Completion tokens: 199
- Mean loss: 2.24318
- Perplexity: 9.42326
- Token accuracy: 0.5729

### javascript — sample 3

- Record: `cc62a70b803e50f71e77671d12ba2abf`
- Source: `lib/shared.js`
- Completion tokens: 126
- Mean loss: 2.12089
- Perplexity: 8.33859
- Token accuracy: 0.5317

### javascript — sample 4

- Record: `71ab9c52c49fddace9d33d9cbe5ed10d`
- Source: `src/js/plugins/youtube.js`
- Completion tokens: 84
- Mean loss: 2.94959
- Perplexity: 19.09807
- Token accuracy: 0.3571

### javascript — sample 5

- Record: `e2d7c5c2bc73b291d0438608c40a1f1b`
- Source: `src/resolver.js`
- Completion tokens: 65
- Mean loss: 1.02304
- Perplexity: 2.78165
- Token accuracy: 0.8000

### javascript — sample 6

- Record: `7e82a51b4a72a68932535886dc014e75`
- Source: `lib/shared-mongo-connections.js`
- Completion tokens: 66
- Mean loss: 1.76272
- Perplexity: 5.82829
- Token accuracy: 0.5909

### javascript — sample 7

- Record: `71ee8117ce2a0834def787b277a2c0a6`
- Source: `src/util/WWMath.js`
- Completion tokens: 39
- Mean loss: 1.45224
- Perplexity: 4.27268
- Token accuracy: 0.5897

### javascript — sample 8

- Record: `7249ecbfd564d411fa9819b4a241d5f9`
- Source: `lib/functions/authenticate.js`
- Completion tokens: 124
- Mean loss: 0.75732
- Perplexity: 2.13254
- Token accuracy: 0.8468

## Integrity

- Tokenizer SHA-256: `0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120`
- Manifest SHA-256: `f0b84da44a2fc02f5a03f32867515eaeb3daf931d77bdbe48d1fe83036e1cbf8`
- Peak allocated VRAM: 141.00 MB
- Training data modified: No
- Checkpoints modified: No
- Test partition opened: No
