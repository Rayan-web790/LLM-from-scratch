# Phase 6B — Validation-Set Generation Diagnostic

**Status:** Validation-only inference experiment (not a correctness score)

**Checkpoint:** `C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT\checkpoints\polycodegpt_first_pass\step_008130.pt` (step 8130)

**Tokenizer:** v1.1.0 (sha256 `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`)

**Selection seed:** 42

**Selected:** 20 (10 python, 10 javascript)

**Distinct record hashes:** 20

**Skipped during selection:** 0 records

**Generation settings:** greedy (temperature=0), max_new_tokens=128

**Data access:** valid.jsonl only.

**Diagnostic measures (definitions):**

- `prefix_match_tokens`: number of leading generated tokens equal to the reference tokens before the first mismatch.

- `exact_continuation_match`: true only when the generated token sequence equals the reference token sequence (both non-empty).

- `token_overlap_jaccard`: |set(generated) ∩ set(reference)| / |set(generated) ∪ set(reference)| over token IDs.

- Empty/nearly empty: <=2 generated tokens before EOS/limit.

- Repetitive: a whitespace 4-gram repeated more than 3 times.

- All of the above are diagnostic token measures, NOT programming-correctness scores. Correctness/quality requires human review.

---

## Example 1 — Python (id 10175c86272c4541)

**Prompt tokens (A):** 13 | **Reference code tokens (A):** 90

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Adds job to neutron context for use later.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def add_job(self, job):
        """Adds job to neutron context for use later."""
        if job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and job.get('job') and
```

**Reference code:**

```
def add_job_to_context(context, job_id):
    """Adds job to neutron context for use later."""
    db_job = db_api.async_transaction_find(
        context, id=job_id, scope=db_api.ONE)
    if not db_job:
        return
    context.async_job = {"job": v._make_job_dict(db_job)}
```

**Diagnostic:** prefix_match=4 exact=False jaccard=0.4038

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 90 | **Prefix tokens:** 45 | **Reference continuation tokens:** 45 | **Prompt tokens:** 58

**Prompt:**

```
<|lang:python|><|docstring|>Adds job to neutron context for use later.<|code|>def add_job_to_context(context, job_id):
    """Adds job to neutron context for use later."""
    db_job = db_api.async_transaction_find(
        context,
```

**Generated (128 tokens, TOKEN LIMIT):**

```

        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_id,
        job_
```

**Reference continuation:**

```
 id=job_id, scope=db_api.ONE)
    if not db_job:
        return
    context.async_job = {"job": v._make_job_dict(db_job)}
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.129

---

## Example 2 — Python (id 809acd292185d9ba)

**Prompt tokens (A):** 14 | **Reference code tokens (A):** 99

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Checks whether there are within-group edges or not.<|code|>
```

**Generated (76 tokens, EOS):**

```
def check_edge_edges(self, edge_edges):
        """Checks whether there are within-group edges or not."""
        if edge_edges is None:
            edge_edges = self.get_edge_edges(edge_edges)
        else:
            edge_edges = self.get_edge_edges(edge_edges)
        return edge_edges
```

**Reference code:**

```
def has_edge_within_group(self, group):
        """
        Checks whether there are within-group edges or not.
        """
        assert group in self.nodes.keys(),\
            "{0} not one of the group of nodes".format(group)
        nodelist = self.nodes[group]
        for n1, n2 in self.simplified_edges():
            if n1 in nodelist and n2 in nodelist:
                return True
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.3913

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 99 | **Prefix tokens:** 49 | **Reference continuation tokens:** 50 | **Prompt tokens:** 63

**Prompt:**

```
<|lang:python|><|docstring|>Checks whether there are within-group edges or not.<|code|>def has_edge_within_group(self, group):
        """
        Checks whether there are within-group edges or not.
        """
        assert group in self.nodes.keys(),\\
            "{0} not one of the group
```

**Generated (15 tokens, EOS):**

```
 {1}".format(group))
        return self.nodes.keys()
```

**Reference continuation:**

```
 of nodes".format(group)
        nodelist = self.nodes[group]
        for n1, n2 in self.simplified_edges():
            if n1 in nodelist and n2 in nodelist:
                return True
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.225

---

## Example 3 — Python (id 4c13b653596e9226)

**Prompt tokens (A):** 18 | **Reference code tokens (A):** 134

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Convert config dict to arguments list.

    :param config: Configuration dict.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def _parse_config(config):
    """Convert config dict to arguments list.

    :param config: Configuration dict.
    """
    config = config.get('config')
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
    config.update(config)
   
```

**Reference code:**

```
def config_to_args(config):
    """Convert config dict to arguments list.

    :param config: Configuration dict.
    """
    result = []

    for key, value in iteritems(config):
        if value is False:
            continue

        key = '--{0}'.format(key.replace('_', '-'))

        if isinstance(value, (list, set, tuple)):
            for item in value:
                result.extend((key, smart_str(item)))
        elif value is not True:
            result.extend((key, smart_str(value)))
        else:
            result.append(key)

    return tuple(result)
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2727

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 134 | **Prefix tokens:** 64 | **Reference continuation tokens:** 70 | **Prompt tokens:** 82

**Prompt:**

```
<|lang:python|><|docstring|>Convert config dict to arguments list.

    :param config: Configuration dict.<|code|>def config_to_args(config):
    """Convert config dict to arguments list.

    :param config: Configuration dict.
    """
    result = []

    for key, value in iteritems(config):
        if value is False:
            continue

        key = '--{0}'.format(key.replace('_',
```

**Generated (12 tokens, EOS):**

```
 '_'))
        result.append(key)

    return result
```

**Reference continuation:**

```
 '-'))

        if isinstance(value, (list, set, tuple)):
            for item in value:
                result.extend((key, smart_str(item)))
        elif value is not True:
            result.extend((key, smart_str(value)))
        else:
            result.append(key)

    return tuple(result)
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.2381

---

## Example 4 — Python (id 9283f15ff937a5bd)

**Prompt tokens (A):** 249 | **Reference code tokens (A):** 343

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Helper function that makes an HTTP POST request to the given firebase
    endpoint. Timeout is 60 seconds.
    `url`: The full URL of the firebase endpoint (DSN appended.)
    `data`: JSON serializable dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The returning value is a Python dict deserialized by the JSON decoder. However,
    if the status code is not 2x or 403, an requests.HTTPError is raised.

    connection = connection_pool.get_available_connection()
    response = make_put_request('http://firebase.localhost/users/',
       '{"Ozgur Vatansever"}', {'X_FIREBASE_SOMETHING': 'Hi'}, connection)
    response => {u'name': u'-Inw6zol_2f5ThHwVcSe'} or {'error': 'Permission denied.'}<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def get_available_connection(request, data, params):
    """Helper function that makes an HTTP POST request to the given firebase
    endpoint. Timeout is 60 seconds.
    `url`: The full URL of the firebase endpoint (DSN appended.)
    `data`: JSON serializable dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The
```

**Reference code:**

```
def make_post_request(url, data, params, headers, connection):
    """
    Helper function that makes an HTTP POST request to the given firebase
    endpoint. Timeout is 60 seconds.
    `url`: The full URL of the firebase endpoint (DSN appended.)
    `data`: JSON serializable dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The returning value is a Python dict deserialized by the JSON decoder. However,
    if the status code is not 2x or 403, an requests.HTTPError is raised.

    connection = connection_pool.get_available_connection()
    response = make_put_request('http://firebase.localhost/users/',
       '{"Ozgur Vatansever"}', {'X_FIREBASE_SOMETHING': 'Hi'}, connection)
    response => {u'name': u'-Inw6zol_2f5ThHwVcSe'} or {'error': 'Permission denied.'}
    """
    timeout = getattr(connection, 'timeout')
    response = connection.post(url, data=data, params=params, headers=headers,
                               timeout=timeout)
    if response.ok or response.status_code == 403:
        return response.json() if response.content else None
    else:
        response.raise_for_status()
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.4124

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 343 | **Prefix tokens:** 64 | **Reference continuation tokens:** 279 | **Prompt tokens:** 313

**Prompt:**

```
<|lang:python|><|docstring|>Helper function that makes an HTTP POST request to the given firebase
    endpoint. Timeout is 60 seconds.
    `url`: The full URL of the firebase endpoint (DSN appended.)
    `data`: JSON serializable dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The returning value is a Python dict deserialized by the JSON decoder. However,
    if the status code is not 2x or 403, an requests.HTTPError is raised.

    connection = connection_pool.get_available_connection()
    response = make_put_request('http://firebase.localhost/users/',
       '{"Ozgur Vatansever"}', {'X_FIREBASE_SOMETHING': 'Hi'}, connection)
    response => {u'name': u'-Inw6zol_2f5ThHwVcSe'} or {'error': 'Permission denied.'}<|code|>def make_post_request(url, data, params, headers, connection):
    """
    Helper function that makes an HTTP POST request to the given firebase
    endpoint. Timeout is 60 seconds.
    `url`: The full URL of the firebase endpoint (DSN appended.)
    `data`: JSON serializable
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The returning value is a Python dict deserialized by the JSON decoder. However,
    if the status code is not 2x or 403, an requests.HTTPError is raised.

    connection = connection_pool.get_available_connection()
    response = make_put_request('http://fire
```

**Reference continuation:**

```
 dict that will be stored in the remote storage.
    `params`: Python dict that is appended to the URL like a querystring.
    `headers`: Python dict. HTTP request headers.
    `connection`: Predefined HTTP connection instance. If not given, it
    is supplied by the `decorators.http_connection` function.

    The returning value is a Python dict deserialized by the JSON decoder. However,
    if the status code is not 2x or 403, an requests.HTTPError is raised.

    connection = connection_pool.get_available_connection()
    response = make_put_request('http://firebase.localhost/users/',
       '{"Ozgur Vatansever"}', {'X_FIREBASE_SOMETHING': 'Hi'}, connection)
    response => {u'name': u'-Inw6zol_2f5ThHwVcSe'} or {'error': 'Permission denied.'}
    """
    timeout = getattr(connection, 'timeout')
    response = connection.post(url, data=data, params=params, headers=headers,
                               timeout=timeout)
    if response.ok or response.status_code == 403:
        return response.json() if response.content else None
    else:
        response.raise_for_status()
```

**Diagnostic:** prefix_match=128 exact=False jaccard=0.481

---

## Example 5 — Python (id d71c0523128daf78)

**Prompt tokens (A):** 6 | **Reference code tokens (A):** 59

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Updates Status information<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def update_status(self, status):
        """Updates Status information """
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.status
        status = self.
```

**Reference code:**

```
def refresh(self, accept=MEDIA_TYPE_TAXII_V20):
        """Updates Status information"""
        response = self.__raw = self._conn.get(self.url,
                                               headers={"Accept": accept})
        self._populate_fields(**response)
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2917

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 59 | **Prefix tokens:** 29 | **Reference continuation tokens:** 30 | **Prompt tokens:** 35

**Prompt:**

```
<|lang:python|><|docstring|>Updates Status information<|code|>def refresh(self, accept=MEDIA_TYPE_TAXII_V20):
        """Updates Status information"""
        response =
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 self.get_response(
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            'POST',
            '
```

**Reference continuation:**

```
 self.__raw = self._conn.get(self.url,
                                               headers={"Accept": accept})
        self._populate_fields(**response)
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2

---

## Example 6 — Python (id 951382f37eba1a72)

**Prompt tokens (A):** 12 | **Reference code tokens (A):** 42

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>设置当前 WechatConf 实例<|code|>
```

**Generated (74 tokens, EOS):**

```
def get_user_info(self, user_info):
        """设置当前 WechatConf 实例"""
        if user_info.get('user_info') == 'user_info':
            return self.user_info.get('user_info')
        else:
            return self.user_info.get('user_info')
```

**Reference code:**

```
def conf(self, conf):
        """ 设置当前 WechatConf 实例  """
        self.__conf = conf
        self.__request = WechatRequest(conf=self.__conf)
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.3864

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 42 | **Prefix tokens:** 21 | **Reference continuation tokens:** 21 | **Prompt tokens:** 33

**Prompt:**

```
<|lang:python|><|docstring|>设置当前 WechatConf 实例<|code|>def conf(self, conf):
        """ 设置当前 WechatConf 实例  """
```

**Generated (128 tokens, TOKEN LIMIT):**

```

        conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf.conf = conf.conf
        conf
```

**Reference continuation:**

```

        self.__conf = conf
        self.__request = WechatRequest(conf=self.__conf)
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2667

---

## Example 7 — Python (id bcb69f1095f8de3d)

**Prompt tokens (A):** 193 | **Reference code tokens (A):** 256

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices

        maxval (int): length of the returned list. If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import ubelt as ub
        >>> indices = [0, 1, 4]
        >>> mask = ub.boolmask(indices, maxval=6)
        >>> assert mask == [True, True, False, False, True, False]
        >>> mask = ub.boolmask(indices)
        >>> assert mask == [True, True, False, False, True]<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def checkmask(indices, maxval=None, maxval=None):
    """Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices

        maxval (int): length of the returned list. If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import
```

**Reference code:**

```
def boolmask(indices, maxval=None):
    """
    Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices

        maxval (int): length of the returned list. If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import ubelt as ub
        >>> indices = [0, 1, 4]
        >>> mask = ub.boolmask(indices, maxval=6)
        >>> assert mask == [True, True, False, False, True, False]
        >>> mask = ub.boolmask(indices)
        >>> assert mask == [True, True, False, False, True]
    """
    if maxval is None:
        indices = list(indices)
        maxval = max(indices) + 1
    mask = [False] * maxval
    for index in indices:
        mask[index] = True
    return mask
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.71

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 256 | **Prefix tokens:** 64 | **Reference continuation tokens:** 192 | **Prompt tokens:** 257

**Prompt:**

```
<|lang:python|><|docstring|>Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices

        maxval (int): length of the returned list. If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import ubelt as ub
        >>> indices = [0, 1, 4]
        >>> mask = ub.boolmask(indices, maxval=6)
        >>> assert mask == [True, True, False, False, True, False]
        >>> mask = ub.boolmask(indices)
        >>> assert mask == [True, True, False, False, True]<|code|>def boolmask(indices, maxval=None):
    """
    Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices

        maxval (int): length of the returned list.
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import ubelt as ub
        >>> indices = [0, 1, 4]
        >>> mask = ub.boolmask(indices, maxval=6)
        >>> assert mask == [True, True, False, False, False]
        >>> mask = ub.boolmask(indices, maxval=6)
```

**Reference continuation:**

```
 If not specified
            this is inferred from `indices`

    Note:
        In the future the arg `maxval` may change its name to `shape`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    Example:
        >>> import ubelt as ub
        >>> indices = [0, 1, 4]
        >>> mask = ub.boolmask(indices, maxval=6)
        >>> assert mask == [True, True, False, False, True, False]
        >>> mask = ub.boolmask(indices)
        >>> assert mask == [True, True, False, False, True]
    """
    if maxval is None:
        indices = list(indices)
        maxval = max(indices) + 1
    mask = [False] * maxval
    for index in indices:
        mask[index] = True
    return mask
```

**Diagnostic:** prefix_match=109 exact=False jaccard=0.8701

---

## Example 8 — Python (id 0275febe0672f394)

**Prompt tokens (A):** 21 | **Reference code tokens (A):** 126

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>return a list (matrix-ish) of the key phrases and their ranks<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def get_key_phrases(self):
        """return a list (matrix-ish) of the key phrases and their ranks """
        if self.key_phrases:
            return self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_phrases[self.key_
```

**Reference code:**

```
def rank_kernel (path):
    """
    return a list (matrix-ish) of the key phrases and their ranks
    """
    kernel = []

    if isinstance(path, str):
        path = json_iter(path)

    for meta in path:
        if not isinstance(meta, RankedLexeme):
            rl = RankedLexeme(**meta)
        else:
            rl = meta

        m = mh_digest(map(lambda x: str(x), rl.ids))
        kernel.append((rl, m,))

    return kernel
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.3544

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 126 | **Prefix tokens:** 63 | **Reference continuation tokens:** 63 | **Prompt tokens:** 84

**Prompt:**

```
<|lang:python|><|docstring|>return a list (matrix-ish) of the key phrases and their ranks<|code|>def rank_kernel (path):
    """
    return a list (matrix-ish) of the key phrases and their ranks
    """
    kernel = []

    if isinstance(path, str):
        path = json_iter(path)

    for meta in path:
        if not isinstance(meta
```

**Generated (40 tokens, EOS):**

```
, dict):
            raise TypeError('meta must be a dict')
        return [meta]

    return [
        (
            (
                'The key phrases is a dict')
        ]
    ]
```

**Reference continuation:**

```
, RankedLexeme):
            rl = RankedLexeme(**meta)
        else:
            rl = meta

        m = mh_digest(map(lambda x: str(x), rl.ids))
        kernel.append((rl, m,))

    return kernel
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.1129

---

## Example 9 — Python (id d0b3c04f790809de)

**Prompt tokens (A):** 11 | **Reference code tokens (A):** 72

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Creates a distribution from a specification dict.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def _create_distribution(self, name, **kwargs):
        """Creates a distribution from a specification dict."""
        if name not in self.distributions:
            raise ValueError("The distribution must be a dictionary")
        if name not in self.distributions:
            raise ValueError("The distribution must be a dictionary")
        if name not in self.distributions:
            raise ValueError("The distribution must be a dictionary")
        if name not in self.distributions:
            raise ValueError("The distribution must be a dictionary")
        if name not in self.distributions:
            raise ValueError("The distribution must be a dictionary")
```

**Reference code:**

```
def from_spec(spec, kwargs=None):
        """
        Creates a distribution from a specification dict.
        """
        distribution = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.distributions.distributions,
            kwargs=kwargs
        )
        assert isinstance(distribution, Distribution)
        return distribution
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2881

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 72 | **Prefix tokens:** 36 | **Reference continuation tokens:** 36 | **Prompt tokens:** 47

**Prompt:**

```
<|lang:python|><|docstring|>Creates a distribution from a specification dict.<|code|>def from_spec(spec, kwargs=None):
        """
        Creates a distribution from a specification dict.
        """
        distribution = util.get_object(
            obj=
```

**Generated (10 tokens, EOS):**

```
spec,
            **kwargs
        )
        return distribution
```

**Reference continuation:**

```
spec,
            predefined_objects=tensorforce.core.distributions.distributions,
            kwargs=kwargs
        )
        assert isinstance(distribution, Distribution)
        return distribution
```

**Diagnostic:** prefix_match=3 exact=False jaccard=0.3077

---

## Example 10 — Python (id 8b5fa121f0498b03)

**Prompt tokens (A):** 103 | **Reference code tokens (A):** 301

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:python|><|docstring|>Convert block number to fork name.

        :param block_number: block number
        :type block_number: int
        :return: fork name
        :rtype: str

        Example use::

            >>> block_to_fork(0)
            ...
            "frontier"
            >>> block_to_fork(4370000)
            ...
            "byzantium"
            >>> block_to_fork(4370001)
            ...
            "byzantium"<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
def block_to_fork(self, block_number):
        """Convert block number to fork name.

        :param block_number: block number
        :type block_number: int
        :return: fork name
        :rtype: str

        Example use::

            >>> block_to_fork(4370000)
            ...
            "byzantium"
            >>> block_to_fork(4370001)
            ...
            "byzantium"
        """
        if block_number is None:
            block_number = block_number
        else:
            block_number = block_number
       
```

**Reference code:**

```
def block_to_fork(block_number):
    """ Convert block number to fork name.

        :param block_number: block number
        :type block_number: int
        :return: fork name
        :rtype: str

        Example use::

            >>> block_to_fork(0)
            ...
            "frontier"
            >>> block_to_fork(4370000)
            ...
            "byzantium"
            >>> block_to_fork(4370001)
            ...
            "byzantium"
    """
    forks_by_block = {
        0: "frontier",
        1150000: "homestead",
        # 1920000 Dao 
        2463000: "tangerine_whistle",
        2675000: "spurious_dragon",
        4370000: "byzantium",
        #7280000: "constantinople", # Same Block as petersburg, commented to avoid conflicts
        7280000: "petersburg",
        9999999: "serenity"  # to be replaced after Serenity launch
    }
    fork_names = list(forks_by_block.values())
    fork_blocks = list(forks_by_block.keys())
    return fork_names[bisect(fork_blocks, block_number) - 1]
```

**Diagnostic:** prefix_match=7 exact=False jaccard=0.3409

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 301 | **Prefix tokens:** 64 | **Reference continuation tokens:** 237 | **Prompt tokens:** 167

**Prompt:**

```
<|lang:python|><|docstring|>Convert block number to fork name.

        :param block_number: block number
        :type block_number: int
        :return: fork name
        :rtype: str

        Example use::

            >>> block_to_fork(0)
            ...
            "frontier"
            >>> block_to_fork(4370000)
            ...
            "byzantium"
            >>> block_to_fork(4370001)
            ...
            "byzantium"<|code|>def block_to_fork(block_number):
    """ Convert block number to fork name.

        :param block_number: block number
        :type block_number: int
        :return: fork name
        :rtype: str

        Example use::

            >>> block_to_fork(0)
```

**Generated (128 tokens, TOKEN LIMIT):**

```

            ...
            "byzantium"
            >>> block_to_fork(4370000)
            ...
            "byzantium"
        """
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
    block_number = block_number
```

**Reference continuation:**

```

            ...
            "frontier"
            >>> block_to_fork(4370000)
            ...
            "byzantium"
            >>> block_to_fork(4370001)
            ...
            "byzantium"
    """
    forks_by_block = {
        0: "frontier",
        1150000: "homestead",
        # 1920000 Dao 
        2463000: "tangerine_whistle",
        2675000: "spurious_dragon",
        4370000: "byzantium",
        #7280000: "constantinople", # Same Block as petersburg, commented to avoid conflicts
        7280000: "petersburg",
        9999999: "serenity"  # to be replaced after Serenity launch
    }
    fork_names = list(forks_by_block.values())
    fork_blocks = list(forks_by_block.keys())
    return fork_names[bisect(fork_blocks, block_number) - 1]
```

**Diagnostic:** prefix_match=4 exact=False jaccard=0.213

---

## Example 11 — Javascript (id f633d5a0ebe8596f)

**Prompt tokens (A):** 7 | **Reference code tokens (A):** 70

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>-- private methods:<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function (methods) {
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods = methods || {};
  var methods =
```

**Reference code:**

```
function parseModelAttributes (attrs, we) {
  if (!attrs) return {};

  let attr = {};

  for (let name in attrs) {
    attr[name] = clone(attrs[name]);
    attr[name].type = getModelTypeFromDefinition(attrs[name], we);
  }

  return attr;
}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.1489

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 70 | **Prefix tokens:** 35 | **Reference continuation tokens:** 35 | **Prompt tokens:** 42

**Prompt:**

```
<|lang:javascript|><|docstring|>-- private methods:<|code|>function parseModelAttributes (attrs, we) {
  if (!attrs) return {};

  let attr = {};

  for (let name in attrs) {
    attr[name
```

**Generated (24 tokens, EOS):**

```
] = attrs[name];
    attr[name] = attrs[name];
  }

  return attr;
}
```

**Reference continuation:**

```
] = clone(attrs[name]);
    attr[name].type = getModelTypeFromDefinition(attrs[name], we);
  }

  return attr;
}
```

**Diagnostic:** prefix_match=2 exact=False jaccard=0.4483

---

## Example 12 — Javascript (id 54c2b619847bdd21)

**Prompt tokens (A):** 34 | **Reference code tokens (A):** 87

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Replaces current node in parent‘s children list with passed nodes
@param {AbbreviationNode} node Replacement node or array of nodes<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function(node) {
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node = node.node;
  var node
```

**Reference code:**

```
function() {
			var parent = this.parent;
			var ix = _.indexOf(parent.children, this);
			var items = _.flatten(arguments);
			spliceFn.apply(parent.children, [ix, 1].concat(items));
			
			// update parent
			_.each(items, function(item) {
				item.parent = parent;
			});
		}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.1458

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 87 | **Prefix tokens:** 43 | **Reference continuation tokens:** 44 | **Prompt tokens:** 77

**Prompt:**

```
<|lang:javascript|><|docstring|>Replaces current node in parent‘s children list with passed nodes
@param {AbbreviationNode} node Replacement node or array of nodes<|code|>function() {
			var parent = this.parent;
			var ix = _.indexOf(parent.children, this);
			var items = _.flatten(arguments);
			spliceFn.apply(
```

**Generated (30 tokens, EOS):**

```
this, items);
			if (items) {
				items.push(items);
			}
			return items;
		}
```

**Reference continuation:**

```
parent.children, [ix, 1].concat(items));
			
			// update parent
			_.each(items, function(item) {
				item.parent = parent;
			});
		}
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.3243

---

## Example 13 — Javascript (id 7775625c978a12cb)

**Prompt tokens (A):** 75 | **Reference code tokens (A):** 79

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Center an entity on a given point.
@private
@param {object} game Required for game.entities.get().
@param {integer} entity The id of entity to center.
@param {object} point A point object <code>{"x": 50, "y": 50}</code> on which to center the entity.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function (game, entity, point) {
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this;
        var self = this
```

**Reference code:**

```
function centerEntityOnPoint(game, entity, point) {
  var size = game.entities.getComponent(entity, "size");
  var position = game.entities.addComponent(entity, "position");
  position.x = point.x - (size.width / 2);
  position.y = point.y - (size.height / 2);
}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2326

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 79 | **Prefix tokens:** 39 | **Reference continuation tokens:** 40 | **Prompt tokens:** 114

**Prompt:**

```
<|lang:javascript|><|docstring|>Center an entity on a given point.
@private
@param {object} game Required for game.entities.get().
@param {integer} entity The id of entity to center.
@param {object} point A point object <code>{"x": 50, "y": 50}</code> on which to center the entity.<|code|>function centerEntityOnPoint(game, entity, point) {
  var size = game.entities.getComponent(entity, "size");
  var position = game.entities.addComponent
```

**Generated (128 tokens, TOKEN LIMIT):**

```
(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.getComponent(position);
  var position = game.entities.
```

**Reference continuation:**

```
(entity, "position");
  position.x = point.x - (size.width / 2);
  position.y = point.y - (size.height / 2);
}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.25

---

## Example 14 — Javascript (id 67b73df8d36b6805)

**Prompt tokens (A):** 12 | **Reference code tokens (A):** 59

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Does the subject deserve this pane?<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function(pane) {
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane = pane.get('pane');
  var pane =
```

**Reference code:**

```
function (subject) {
    var n = UI.widgets.formsFor(subject).length
    UI.log.debug('Form pane: forms for ' + subject + ': ' + n)
    if (!n) return null
    return '' + n + ' forms'
  }
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.2619

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 59 | **Prefix tokens:** 29 | **Reference continuation tokens:** 30 | **Prompt tokens:** 41

**Prompt:**

```
<|lang:javascript|><|docstring|>Does the subject deserve this pane?<|code|>function (subject) {
    var n = UI.widgets.formsFor(subject).length
    UI.log.debug('Form pane
```

**Generated (128 tokens, TOKEN LIMIT):**

```
: %s', subject).then(function (e) {
        return e.message;
    });

    // If the subject deserve this pane?
    if (subject.id === 'pane') {
        return e.message;
    }

    // If the subject deserve this pane?
    if (subject.id === 'pane') {
        return e.message;
    }

    // If the subject deserve this pane?
    if (subject.id === 'pane') {
        return e.message;
    }

    // If the subject deserve this pane?
    if (
```

**Reference continuation:**

```
: forms for ' + subject + ': ' + n)
    if (!n) return null
    return '' + n + ' forms'
  }
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.16

---

## Example 15 — Javascript (id 915cd810e4549173)

**Prompt tokens (A):** 87 | **Reference code tokens (A):** 129

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Processes the conditions within the provided template using the provided
condition matcher regexp.

@param {string} template The template to process.
@param {Object<string, *>} variables The variables that should be available
to the predicate expressions in the conditions.
@param {RegExp} matcher The regexp to use to match the conditions in the
template.
@return {string} Processed template.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function process(template, variables) {
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
  var result = {};
```

**Reference code:**

```
function processCondition(template, variables, matcher) {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    let matchedCondition = template.match(matcher);
    if (!matchedCondition) {
      break;
    }

    let fragment = matchedCondition[0];
    let conditionCode = matchedCondition[1];

    let testResult = evaluateCondition(conditionCode, variables);
    let replaceWith = testResult
      ? matchedCondition[2]
      : matchedCondition[3] || '';

    template = template.replace(fragment, replaceWith);
  }

  return template;
}
```

**Diagnostic:** prefix_match=2 exact=False jaccard=0.1471

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 129 | **Prefix tokens:** 64 | **Reference continuation tokens:** 65 | **Prompt tokens:** 151

**Prompt:**

```
<|lang:javascript|><|docstring|>Processes the conditions within the provided template using the provided
condition matcher regexp.

@param {string} template The template to process.
@param {Object<string, *>} variables The variables that should be available
to the predicate expressions in the conditions.
@param {RegExp} matcher The regexp to use to match the conditions in the
template.
@return {string} Processed template.<|code|>function processCondition(template, variables, matcher) {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    let matchedCondition = template.match(matcher);
    if (!matchedCondition) {
      break;
    }

    let fragment = matchedCondition[0];
   
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(fragment);
    }
    if (fragment.length === 0) {
      fragment.push(
```

**Reference continuation:**

```
 let conditionCode = matchedCondition[1];

    let testResult = evaluateCondition(conditionCode, variables);
    let replaceWith = testResult
      ? matchedCondition[2]
      : matchedCondition[3] || '';

    template = template.replace(fragment, replaceWith);
  }

  return template;
}
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.1429

---

## Example 16 — Javascript (id 0ffca5b64ba58f80)

**Prompt tokens (A):** 62 | **Reference code tokens (A):** 132

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>逻辑删除情况下加工查询对象
@param {Object} name - 模型名称
@param {Object} query - 查询对象
@param {Object} opts - 适配器配置<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function(name, query, opts) {
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result = {};
    var result =
```

**Reference code:**

```
function tombstoneQuery(name, query, opts) {
    const tombstoneKey = opts && (typeof opts.tombstoneKeyGetter === 'function') ? opts.tombstoneKeyGetter(name) : null;
    if (!tombstoneKey) return query;

    const cond = { [tombstoneKey]: { $ne: true } };
    if (query._conditions.$and && Array.isArray(query._conditions.$and)) {
        query._conditions.$and.push(cond);
    } else {
        Object.assign(query._conditions, cond);
    }
    return query;
}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.1613

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 132 | **Prefix tokens:** 64 | **Reference continuation tokens:** 68 | **Prompt tokens:** 126

**Prompt:**

```
<|lang:javascript|><|docstring|>逻辑删除情况下加工查询对象
@param {Object} name - 模型名称
@param {Object} query - 查询对象
@param {Object} opts - 适配器配置<|code|>function tombstoneQuery(name, query, opts) {
    const tombstoneKey = opts && (typeof opts.tombstoneKeyGetter === 'function') ? opts.tombstoneKeyGetter(name) : null;
    if (!tombstoneKey) return query;

    const cond =
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 opts.cond || (typeof opts.tombstoneKeyGetter === 'function') ? opts.tombstoneKeyGetter(name) : null;
    if (cond) {
        cond = opts.cond || (typeof opts.tombstoneKeyGetter === 'function') ? opts.tombstoneKeyGetter(name) : null;
    }

    const result = {
        'name': name,
        'result': result,
        'result': result,
        'result': result,
        'result': result,
        'result': result,
        'result': result,
        'result': result,
        '
```

**Reference continuation:**

```
 { [tombstoneKey]: { $ne: true } };
    if (query._conditions.$and && Array.isArray(query._conditions.$and)) {
        query._conditions.$and.push(cond);
    } else {
        Object.assign(query._conditions, cond);
    }
    return query;
}
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.2623

---

## Example 17 — Javascript (id 830219fe9e86cad1)

**Prompt tokens (A):** 6 | **Reference code tokens (A):** 370

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>/* 417<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function(a, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b,
```

**Reference code:**

```
function(module, exports, __webpack_require__) {

	var _curry2 = /*#__PURE__*/__webpack_require__(1);
	
	/**
	 * Takes a list and a predicate and returns a pair of lists with the following properties:
	 *
	 *  - the result of concatenating the two output lists is equivalent to the input list;
	 *  - none of the elements of the first output list satisfies the predicate; and
	 *  - if the second output list is non-empty, its first element satisfies the predicate.
	 *
	 * @func
	 * @memberOf R
	 * @since v0.19.0
	 * @category List
	 * @sig (a -> Boolean) -> [a] -> [[a], [a]]
	 * @param {Function} pred The predicate that determines where the array is split.
	 * @param {Array} list The array to be split.
	 * @return {Array}
	 * @example
	 *
	 *      R.splitWhen(R.equals(2), [1, 2, 3, 1, 2, 3]);   //=> [[1], [2, 3, 1, 2, 3]]
	 */
	
	
	var splitWhen = /*#__PURE__*/_curry2(function splitWhen(pred, list) {
	  var idx = 0;
	  var len = list.length;
	  var prefix = [];
	
	  while (idx < len && !pred(list[idx])) {
	    prefix.push(list[idx]);
	    idx += 1;
	  }
	
	  return [prefix, Array.prototype.slice.call(list, idx)];
	});
	module.exports = splitWhen;

/***/ }
```

**Diagnostic:** prefix_match=2 exact=False jaccard=0.0263

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 370 | **Prefix tokens:** 64 | **Reference continuation tokens:** 306 | **Prompt tokens:** 70

**Prompt:**

```
<|lang:javascript|><|docstring|>/* 417<|code|>function(module, exports, __webpack_require__) {

	var _curry2 = /*#__PURE__*/__webpack_require__(1);
	
	/**
	 * Takes a list and a predicate and returns a pair of lists with the following properties:
	 *
	 *  - the result
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the predicate
	 *  - the result of the
```

**Reference continuation:**

```
 of concatenating the two output lists is equivalent to the input list;
	 *  - none of the elements of the first output list satisfies the predicate; and
	 *  - if the second output list is non-empty, its first element satisfies the predicate.
	 *
	 * @func
	 * @memberOf R
	 * @since v0.19.0
	 * @category List
	 * @sig (a -> Boolean) -> [a] -> [[a], [a]]
	 * @param {Function} pred The predicate that determines where the array is split.
	 * @param {Array} list The array to be split.
	 * @return {Array}
	 * @example
	 *
	 *      R.splitWhen(R.equals(2), [1, 2, 3, 1, 2, 3]);   //=> [[1], [2, 3, 1, 2, 3]]
	 */
	
	
	var splitWhen = /*#__PURE__*/_curry2(function splitWhen(pred, list) {
	  var idx = 0;
	  var len = list.length;
	  var prefix = [];
	
	  while (idx < len && !pred(list[idx])) {
	    prefix.push(list[idx]);
	    idx += 1;
	  }
	
	  return [prefix, Array.prototype.slice.call(list, idx)];
	});
	module.exports = splitWhen;

/***/ }
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.0522

---

## Example 18 — Javascript (id 8d7c97c5f1f682a6)

**Prompt tokens (A):** 31 | **Reference code tokens (A):** 172

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Set options defaults that are used for batter tasks
@param opts Object with options for batter
@returns {*|{}}<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function setOptions(opts) {
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
  opts = opts || {};
 
```

**Reference code:**

```
function getOptions(opts) {
    opts = opts || {};

    // command line arguments override options
    _.extend(opts, argv);
    delete opts.$0;
    delete opts._;

    // these values are used in multiple tasks, so set defaults here
    opts.unitTestCode = opts.unitTestCode || 'test/unit/**/*.js';
    opts.unitTargetCode = opts.unitTargetCode || 'lib/**/*.js';
    opts.testDir = opts.testDir || 'test';
    opts.rootDir = opts.rootDir || process.cwd();
    opts.targetDir = opts.targetDir || (opts.rootDir + '/lib');
    opts.tasksets = _.extend({ 'default': ['lint', 'test'] }, opts.tasksets);

    return opts;
}
```

**Diagnostic:** prefix_match=1 exact=False jaccard=0.1429

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 172 | **Prefix tokens:** 64 | **Reference continuation tokens:** 108 | **Prompt tokens:** 95

**Prompt:**

```
<|lang:javascript|><|docstring|>Set options defaults that are used for batter tasks
@param opts Object with options for batter
@returns {*|{}}<|code|>function getOptions(opts) {
    opts = opts || {};

    // command line arguments override options
    _.extend(opts, argv);
    delete opts.$0;
    delete opts._;

    // these values are used in multiple tasks, so set defaults here
    opts.unitTestCode = opts.unitTest
```

**Generated (128 tokens, TOKEN LIMIT):**

```
Code || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode || opts.unitTestCode ||
```

**Reference continuation:**

```
Code || 'test/unit/**/*.js';
    opts.unitTargetCode = opts.unitTargetCode || 'lib/**/*.js';
    opts.testDir = opts.testDir || 'test';
    opts.rootDir = opts.rootDir || process.cwd();
    opts.targetDir = opts.targetDir || (opts.rootDir + '/lib');
    opts.tasksets = _.extend({ 'default': ['lint', 'test'] }, opts.tasksets);

    return opts;
}
```

**Diagnostic:** prefix_match=2 exact=False jaccard=0.1087

---

## Example 19 — Javascript (id efc4f67b627641e5)

**Prompt tokens (A):** 6 | **Reference code tokens (A):** 364

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>/* 350<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function(err, res) {
  if (err) {
    res = res.toString();
  }
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
  res.push(err);
 
```

**Reference code:**

```
function(module, exports, __webpack_require__) {

	var _checkForMethod = /*#__PURE__*/__webpack_require__(23);
	
	var _curry2 = /*#__PURE__*/__webpack_require__(1);
	
	/**
	 * Creates a new list with the separator interposed between elements.
	 *
	 * Dispatches to the `intersperse` method of the second argument, if present.
	 *
	 * @func
	 * @memberOf R
	 * @since v0.14.0
	 * @category List
	 * @sig a -> [a] -> [a]
	 * @param {*} separator The element to add to the list.
	 * @param {Array} list The list to be interposed.
	 * @return {Array} The new list.
	 * @example
	 *
	 *      R.intersperse('n', ['ba', 'a', 'a']); //=> ['ba', 'n', 'a', 'n', 'a']
	 */
	
	
	var intersperse = /*#__PURE__*/_curry2( /*#__PURE__*/_checkForMethod('intersperse', function intersperse(separator, list) {
	  var out = [];
	  var idx = 0;
	  var length = list.length;
	  while (idx < length) {
	    if (idx === length - 1) {
	      out.push(list[idx]);
	    } else {
	      out.push(list[idx], separator);
	    }
	    idx += 1;
	  }
	  return out;
	}));
	module.exports = intersperse;

/***/ }
```

**Diagnostic:** prefix_match=2 exact=False jaccard=0.0863

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 364 | **Prefix tokens:** 64 | **Reference continuation tokens:** 300 | **Prompt tokens:** 70

**Prompt:**

```
<|lang:javascript|><|docstring|>/* 350<|code|>function(module, exports, __webpack_require__) {

	var _checkForMethod = /*#__PURE__*/__webpack_require__(23);
	
	var _curry2 = /*#__PURE__*/__webpack_require__(1);
	
	/**
	 * Creates a new
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 object with the given object.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {object} obj The object to create.
	 * @param {
```

**Reference continuation:**

```
 list with the separator interposed between elements.
	 *
	 * Dispatches to the `intersperse` method of the second argument, if present.
	 *
	 * @func
	 * @memberOf R
	 * @since v0.14.0
	 * @category List
	 * @sig a -> [a] -> [a]
	 * @param {*} separator The element to add to the list.
	 * @param {Array} list The list to be interposed.
	 * @return {Array} The new list.
	 * @example
	 *
	 *      R.intersperse('n', ['ba', 'a', 'a']); //=> ['ba', 'n', 'a', 'n', 'a']
	 */
	
	
	var intersperse = /*#__PURE__*/_curry2( /*#__PURE__*/_checkForMethod('intersperse', function intersperse(separator, list) {
	  var out = [];
	  var idx = 0;
	  var length = list.length;
	  while (idx < length) {
	    if (idx === length - 1) {
	      out.push(list[idx]);
	    } else {
	      out.push(list[idx], separator);
	    }
	    idx += 1;
	  }
	  return out;
	}));
	module.exports = intersperse;

/***/ }
```

**Diagnostic:** prefix_match=0 exact=False jaccard=0.0873

---

## Example 20 — Javascript (id b1467f9e696021a8)

**Prompt tokens (A):** 48 | **Reference code tokens (A):** 73

### Experiment A — Docstring-to-code

**Prompt:**

```
<|lang:javascript|><|docstring|>Set data to the given element.
@param {Element} element - The target element.
@param {string} name - The data key to set.
@param {string} data - The data value.<|code|>
```

**Generated (128 tokens, TOKEN LIMIT):**

```
function setData(element, name, data) {
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element.data;
  var data = element
```

**Reference code:**

```
function setData(element, name, data) {
  if (isObject(data)) {
    element[name] = data;
  } else if (element.dataset) {
    element.dataset[name] = data;
  } else {
    element.setAttribute("data-".concat(toParamCase(name)), data);
  }
}
```

**Diagnostic:** prefix_match=12 exact=False jaccard=0.4

### Experiment B — Code-prefix continuation

**Code tokens (EOS excluded):** 73 | **Prefix tokens:** 36 | **Reference continuation tokens:** 37 | **Prompt tokens:** 84

**Prompt:**

```
<|lang:javascript|><|docstring|>Set data to the given element.
@param {Element} element - The target element.
@param {string} name - The data key to set.
@param {string} data - The data value.<|code|>function setData(element, name, data) {
  if (isObject(data)) {
    element[name] = data;
  } else if (element.dataset)
```

**Generated (128 tokens, TOKEN LIMIT):**

```
 {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
    element[name] = data;
  } else if (element.dataset) {
   
```

**Reference continuation:**

```
 {
    element.dataset[name] = data;
  } else {
    element.setAttribute("data-".concat(toParamCase(name)), data);
  }
}
```

**Diagnostic:** prefix_match=3 exact=False jaccard=0.4375

---

## Summary (automated diagnostics)

- Examples: 20

- Repetitive generations: A=14, B=12 (4-gram repeated >3 times; heuristic).

- Empty or nearly empty continuations (<=2 tokens): A=0, B=0.

- Structurally plausible but semantically incorrect code: not automatically scored; manual review of the generated outputs above is required.

- Generations resembling their validation references: mean token-overlap Jaccard A=0.2802, B=0.2659; exact continuation matches A=0, B=0. These are diagnostic token measures, not correctness labels.

- Experiment A vs B: A stopped at EOS 2 times, B 6 times; A empty=0/repetitive=14/exact=0, B empty=0/repetitive=12/exact=0. Differences indicate whether docstring-to-code or code-continuation prompting is the harder setting for this checkpoint.

- No generation is labelled correct merely because its tokens resemble the reference.
