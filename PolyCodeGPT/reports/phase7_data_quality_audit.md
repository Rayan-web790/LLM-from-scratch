# Phase 7 — PolyCodeGPT Training Data Quality Audit

Read-only audit. No dataset, checkpoint, tokenizer, or configuration was modified.
No training, GPU generation, or held-out test evaluation was performed.

## 1. Data boundaries

Allowed and inspected:
- `data/processed/polycodegpt_corpus/train.jsonl`
- `data/processed/polycodegpt_corpus/valid.jsonl`

Not accessed:
- Any test JSONL or binary data, including `test.jsonl` and `test.bin`.
- Any held-out test results or metrics.

All record identifiers below are 16-character SHA-256 hashes of stable record IDs; repository names, paths, URLs, and personal information are not exposed.

## 2. Dataset inspection

### 2.1 Record counts

| Split | Python | JavaScript | Total |
|---|---:|---:|---:|
| train | 290,463 | 102,169 | 392,632 |
| valid | 14,671 | 6,294 | 20,965 |
| Total | 305,134 | 108,463 | 413,597 |

Every record has a non-empty docstring and non-empty code (0 missing/empty in both splits).
All records have unique `record_id` values and unique `code_sha256` values.
There are 0 code duplicates across train and validation splits.

### 2.2 Character and tokenizer lengths

Tokenizer lengths are measured with the existing tokenizer v1.1.0 on the raw content (not the CodeSearchNet whitespace token lists stored in the corpus).

| Split | Field | min | p10 | median | p90 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| train | docstring chars | 1 | 31.0 | 137.0 | 606.0 | 46934 | 273.8 |
| train | code chars | 52 | 235.0 | 612.0 | 2040.0 | 889829 | 1018.6 |
| train | docstring tokens | 1 | 7.0 | 33.0 | 145.0 | 10276 | 65.9 |
| train | code tokens | 25 | 63.0 | 158.0 | 520.0 | 624839 | 262.5 |
| valid | docstring chars | 1 | 30.0 | 143.0 | 725.0 | 21748 | 316.5 |
| valid | code chars | 69 | 228.0 | 615.0 | 2141.6 | 86202 | 1081.6 |
| valid | docstring tokens | 1 | 7.0 | 34.0 | 177.0 | 5531 | 79.9 |
| valid | code tokens | 26 | 62.0 | 161.0 | 573.0 | 20356 | 286.7 |

Unusually short or long code (tokenizer tokens):

| Split | <=24 tokens | p1 | p99 | records >= p99 |
|---|---:|---:|---:|---:|
| train | 0 | 40 | 1610 | 3929 |
| valid | 0 | 39 | 1998 | 211 |

No record has 24 or fewer code tokens, because the corpus builder already enforced a minimum code length. The p99 tail (long files/bundles) is retained and is part of the training stream.

### 2.3 Non-ASCII and multilingual docstrings

Non-ASCII docstrings: 10,565 of 413,597 (2.55%).

| Language | Non-ASCII docstrings |
|---|---:|
| python | 5,020 |
| javascript | 5,545 |

Unicode scripts observed include CJK, Hangul, Cyrillic, Arabic, Greek, Thai, Devanagari, and many mathematical/symbol ranges. Many of these are genuine multilingual documentation or docstrings containing formulas, paths, and identifiers; they are not automatically invalid.

### 2.4 Repeated or repetitive code fragments

| Indicator (heuristic) | Python | JavaScript |
|---|---:|---:|
| Repeated non-blank line ratio > 0.2 | 6,931 | 22,507 |
| 4-gram token repeat count > 5 | 2,716 | 1,971 |
| Comment/blank-heavy code (comment ratio > 0.5) | 248 | 973 |
| Repeated docstring values (duplicate docstrings) | 21,676 extra occurrences across 21,676 duplicate groups | |

The most common repeated docstring is `"Auto Generated Code"` (1,779 occurrences), followed by `"Return a json dictionary representing this model."` (286) and `"/*"` (125). These are machine-generated or stub docstrings and are a plausible source of weak description/code alignment, but they are not by themselves proof of bad training data.

### 2.5 Bundles, minified files, wrappers, and incomplete fragments

| Indicator (heuristic) | Python | JavaScript | Notes |
|---|---:|---:|---|
| Minified-like (long average lines or many long lines) | 720 | 281 | heuristic only |
| Bundle/module markers (webpack, sourceMappingURL, many functions) | 1,387 | 2,184 | heuristic only |
| Wrapper/decorator-heavy code | 16,765 | 6,020 | heuristic only |
| Bracket/parenthesis imbalance or non-standard ending | 133,708 | 1,689 | high false-positive rate; see note |

Note on the "incomplete" indicator: it is based on simple bracket balance and final character checks and has a high false-positive rate (for example, strings, comments, and multiline literals can make a complete function look unbalanced). It should be treated as a triage hint, not a validity proof. The bundle and minified indicators are more reliable.

## 3. Deterministic sample review

Selection: seed 42, stratified by code-token length (tertiles) so ordinary and unusual lengths are represented.

| Group | Requested | Selected |
|---|---:|---:|
| Python training | 30 | 30 |
| JavaScript training | 30 | 30 |
| Python validation | 15 | 15 |
| JavaScript validation | 15 | 15 |
| Total | 90 | 90 |

Each sample below gives the stable record hash, language, character/tokenizer lengths, a brief alignment assessment, and truncated docstring/code excerpts.
Assessments are qualitative judgments supported by machine indicators; they are not proof of semantic correctness.

### 3.1 Python training samples

#### train_python-01 — `221ec4bc67a8027b`
- Language: python; split: train
- Lengths: docstring 329 chars / 67 tokens; code 478 chars / 111 tokens
- Machine indicators: function names ['and', 'from_function', 'to']; names in docstring ['and', 'to']; token overlap 0.765; comment ratio 0.0; repeated-line ratio 0.091; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Create a new instance from a function. Gets the code object from
        the function and passes it and any other specified parameters to
        :meth:`from_code`.

        Arguments:
            f(function): The function to get the code o
~~~
- Code excerpt:
~~~text
def from_function(cls, f, *args, **kwargs):
        """
        Create a new instance from a function. Gets the code object from
        the function and passes it and any other specified parameters to
        :meth:`from_code`.

        Arguments:
            f(function): The function to get the code object from.

   
~~~

#### train_python-02 — `7eab2b3ce98b2f1c`
- Language: python; split: train
- Lengths: docstring 12 chars / 4 tokens; code 236 chars / 58 tokens
- Machine indicators: function names ['byte_to_channel']; names in docstring []; token overlap 0.182; comment ratio 0.0; repeated-line ratio 0.125; bundle []; minified False
- Assessment: Partial alignment — docstring is only a return-type note, but the code is a straightforward accessor.
- Docstring excerpt:
~~~text
:return: int
~~~
- Code excerpt:
~~~text
def byte_to_channel(self, byte):
        """
        :return: int
        """
        assert isinstance(byte, int)
        channels = self.byte_to_channels(byte)
        self.needs_one_channel(channels)
        return channels[0]
~~~

#### train_python-03 — `33a0907416543ebc`
- Language: python; split: train
- Lengths: docstring 21 chars / 6 tokens; code 148 chars / 45 tokens
- Machine indicators: function names ['_stripped']; names in docstring []; token overlap 0.2; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Strip falsey entries.
~~~
- Code excerpt:
~~~text
def _stripped(d):
    '''Strip falsey entries.'''
    ret = {}
    for k, v in six.iteritems(d):
        if v:
            ret[k] = v
    return ret
~~~

#### train_python-04 — `e01873277cbff7ba`
- Language: python; split: train
- Lengths: docstring 57 chars / 12 tokens; code 344 chars / 75 tokens
- Machine indicators: function names ['is_valid']; names in docstring []; token overlap 0.385; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Returns a Client object if this is a valid OAuth request.
~~~
- Code excerpt:
~~~text
def is_valid(self):
        """Returns a Client object if this is a valid OAuth request."""

        try:
            request = self.get_oauth_request()
            client = self.get_client(request)
            params = self._server.verify_request(request, client, None)
        except Exception as e:
            raise 
~~~

#### train_python-05 — `257cb5983f5b760b`
- Language: python; split: train
- Lengths: docstring 55 chars / 14 tokens; code 269 chars / 72 tokens
- Machine indicators: function names ['rich_presence']; names in docstring []; token overlap 0.368; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Contains Rich Presence key-values

        :rtype: dict
~~~
- Code excerpt:
~~~text
def rich_presence(self):
        """Contains Rich Presence key-values

        :rtype: dict
        """
        kvs = self.get_ps('rich_presence')
        data = {}

        if kvs:
            for kv in kvs:
                data[kv.key] = kv.value

        return data
~~~

#### train_python-06 — `f6419795b27b55b0`
- Language: python; split: train
- Lengths: docstring 48 chars / 10 tokens; code 247 chars / 70 tokens
- Machine indicators: function names ['access']; names in docstring ['access']; token overlap 0.353; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
TextureAccess: The actual access to the texture.
~~~
- Code excerpt:
~~~text
def access(self):
        """TextureAccess: The actual access to the texture."""
        access = ffi.new('int *')
        check_int_err(lib.SDL_QueryTexture(self._ptr, ffi.NULL, access, ffi.NULL, ffi.NULL))
        return TextureAccess(access[0])
~~~

#### train_python-07 — `d5739649f6f91dbf`
- Language: python; split: train
- Lengths: docstring 38 chars / 6 tokens; code 167 chars / 61 tokens
- Machine indicators: function names ['p_expression_eql']; names in docstring []; token overlap 0.25; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
expression : expression EQL expression
~~~
- Code excerpt:
~~~text
def p_expression_eql(self, p):
        'expression : expression EQL expression'
        p[0] = Eql(p[1], p[3], lineno=p.lineno(1))
        p.set_lineno(0, p.lineno(1))
~~~

#### train_python-08 — `0aa57e52c1c8ac13`
- Language: python; split: train
- Lengths: docstring 230 chars / 54 tokens; code 479 chars / 121 tokens
- Machine indicators: function names ['create_skel']; names in docstring []; token overlap 0.651; comment ratio 0.0; repeated-line ratio 0.091; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Create the skeleton of a dictionary or JSON file

    A dictionary is returned that contains the "molssi_bse_schema"
    key and other required keys, depending on the role

    role can be either 'component', 'element', or 'table'
~~~
- Code excerpt:
~~~text
def create_skel(role):
    '''
    Create the skeleton of a dictionary or JSON file

    A dictionary is returned that contains the "molssi_bse_schema"
    key and other required keys, depending on the role

    role can be either 'component', 'element', or 'table'
    '''

    role = role.lower()
    if not role in _s
~~~

#### train_python-09 — `b7a34eaaaeb1a69a`
- Language: python; split: train
- Lengths: docstring 49 chars / 10 tokens; code 233 chars / 57 tokens
- Machine indicators: function names ['_initialize_progressbar']; names in docstring []; token overlap 0.357; comment ratio 0.0; repeated-line ratio 0.167; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Initialize the progressbar.
        :return: None
~~~
- Code excerpt:
~~~text
def _initialize_progressbar(self):
        """
        Initialize the progressbar.
        :return: None
        """

        self._progressbar = progressbar.ProgressBar(widgets=Analysis._PROGRESS_WIDGETS, maxval=10000 * 100).start()
~~~

#### train_python-10 — `b7d05b8c522fede2`
- Language: python; split: train
- Lengths: docstring 219 chars / 35 tokens; code 494 chars / 115 tokens
- Machine indicators: function names ['_create_objective', 'and']; names in docstring ['and']; token overlap 0.618; comment ratio 0.067; repeated-line ratio 0.067; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Parameters
        ----------
        m, n : int
            Dimensions that of solution matrix
        Returns the objective function and a variable representing the
        solution to the convex optimization problem.
~~~
- Code excerpt:
~~~text
def _create_objective(self, m, n):
        """
        Parameters
        ----------
        m, n : int
            Dimensions that of solution matrix
        Returns the objective function and a variable representing the
        solution to the convex optimization problem.
        """
        # S is the completed matr
~~~

#### train_python-11 — `77a9684dd3a06ce4`
- Language: python; split: train
- Lengths: docstring 530 chars / 123 tokens; code 818 chars / 199 tokens
- Machine indicators: function names ['delete_role']; names in docstring []; token overlap 0.84; comment ratio 0.0; repeated-line ratio 0.05; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Delete an existing role by the given name.

        If the role does not exist, a 404 is returned.

        Supported methods:
            DELETE: /{mount_point}/roles/{name}. Produces: 204 (empty body)

        :param name: the name of the
~~~
- Code excerpt:
~~~text
def delete_role(self, name, mount_point=DEFAULT_MOUNT_POINT):
        """Delete an existing role by the given name.

        If the role does not exist, a 404 is returned.

        Supported methods:
            DELETE: /{mount_point}/roles/{name}. Produces: 204 (empty body)

        :param name: the name of the role t
~~~

#### train_python-12 — `60fd8d6f72c66754`
- Language: python; split: train
- Lengths: docstring 21 chars / 4 tokens; code 539 chars / 131 tokens
- Machine indicators: function names ['on_data']; names in docstring []; token overlap 0.097; comment ratio 0.133; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Partial alignment — vague one-line docstring, but the code clearly handles received data.
- Docstring excerpt:
~~~text
Handle received data.
~~~
- Code excerpt:
~~~text
async def on_data(self, data):
        """ Handle received data. """
        self._receive_buffer += data

        # Schedule new timeout event.
        if self._ping_checker_handle:
            self._ping_checker_handle.cancel()

        # create a task for the ping checker
        self._ping_checker_handle = self.eve
~~~

#### train_python-13 — `48e0ecb1372fb194`
- Language: python; split: train
- Lengths: docstring 550 chars / 149 tokens; code 739 chars / 208 tokens
- Machine indicators: function names ['generate_aead_simple']; names in docstring []; token overlap 0.889; comment ratio 0.0; repeated-line ratio 0.067; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Generate AEAD block from data for a specific key in a single step
        (without using the YubiHSM internal buffer).

        @param nonce: The nonce to use when creating the AEAD
        @param key_handle: The key handle that can encrypt
~~~
- Code excerpt:
~~~text
def generate_aead_simple(self, nonce, key_handle, data):
        """
        Generate AEAD block from data for a specific key in a single step
        (without using the YubiHSM internal buffer).

        @param nonce: The nonce to use when creating the AEAD
        @param key_handle: The key handle that can encrypt da
~~~

#### train_python-14 — `cc9294f556dab95f`
- Language: python; split: train
- Lengths: docstring 96 chars / 19 tokens; code 768 chars / 177 tokens
- Machine indicators: function names ['is', 'notify_dispatch_request']; names in docstring ['is']; token overlap 0.233; comment ratio 0.286; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Will be called when a context passed to a hooked function is about
        to dispatch a request
~~~
- Code excerpt:
~~~text
def notify_dispatch_request(self, request):
        # type: (Request) -> None
        """Will be called when a context passed to a hooked function is about
        to dispatch a request"""
        if isinstance(request, Put) and request.path[0] == self.mri:
            # This means the context we were passed has just m
~~~

#### train_python-15 — `57835cfe17de642d`
- Language: python; split: train
- Lengths: docstring 91 chars / 20 tokens; code 514 chars / 125 tokens
- Machine indicators: function names ['create_person']; names in docstring []; token overlap 0.375; comment ratio 0.0; repeated-line ratio 0.083; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Create a new person
        :param person: is the new person that the client wants to crete
~~~
- Code excerpt:
~~~text
def create_person(self, person):
        """
        Create a new person
        :param person: is the new person that the client wants to crete
        """
        self._validate_subscriber_id(person.surrogate_id)

        url = "/notification/v1/person"
        response = NWS_DAO().postURL(
            url, self._wri
~~~

#### train_python-16 — `3a340e0174c06961`
- Language: python; split: train
- Lengths: docstring 59 chars / 11 tokens; code 484 chars / 125 tokens
- Machine indicators: function names ['get_environ']; names in docstring []; token overlap 0.333; comment ratio 0.167; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Create WSGI environ entries to be merged into each request.
~~~
- Code excerpt:
~~~text
def get_environ(self, sock):
        """Create WSGI environ entries to be merged into each request."""
        cipher = sock.cipher()
        ssl_environ = {
            "wsgi.url_scheme": "https",
            "HTTPS": "on",
            'SSL_PROTOCOL': cipher[1],
            'SSL_CIPHER': cipher[0]
##            SSL_VE
~~~

#### train_python-17 — `15393c225ab67b41`
- Language: python; split: train
- Lengths: docstring 436 chars / 89 tokens; code 613 chars / 132 tokens
- Machine indicators: function names ['before', 'before_app_websocket']; names in docstring ['before', 'before_app_websocket']; token overlap 0.844; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Add a before request websocket to the App.

        This is designed to be used as a decorator, and has the same arguments
        as :meth:`~quart.Quart.before_websocket`. It applies to all requests to the
        app this blueprint is reg
~~~
- Code excerpt:
~~~text
def before_app_websocket(self, func: Callable) -> Callable:
        """Add a before request websocket to the App.

        This is designed to be used as a decorator, and has the same arguments
        as :meth:`~quart.Quart.before_websocket`. It applies to all requests to the
        app this blueprint is registered o
~~~

#### train_python-18 — `cc2288dcb47ed33c`
- Language: python; split: train
- Lengths: docstring 272 chars / 58 tokens; code 611 chars / 147 tokens
- Machine indicators: function names ['_load_from_hdx']; names in docstring []; token overlap 0.711; comment ratio 0.056; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Helper method to load the HDX object given by identifier from HDX

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field (str): HDX object identifier

        Returns:
            bo
~~~
- Code excerpt:
~~~text
def _load_from_hdx(self, object_type, id_field):
        # type: (str, str) -> bool
        """Helper method to load the HDX object given by identifier from HDX

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field (str): HDX object identifier

        Returns:
~~~

#### train_python-19 — `b32df2e8867bb16b`
- Language: python; split: train
- Lengths: docstring 296 chars / 68 tokens; code 549 chars / 149 tokens
- Machine indicators: function names ['get_host_health_fan_sensors']; names in docstring []; token overlap 0.718; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Get the health Fan Sensor Report.

        :param: the data to retrieve from the server, defaults to None.
        :returns: the dictionary containing the fan sensor information.
        :raises: IloConnectionError if failed connecting to t
~~~
- Code excerpt:
~~~text
def get_host_health_fan_sensors(self, data=None):
        """Get the health Fan Sensor Report.

        :param: the data to retrieve from the server, defaults to None.
        :returns: the dictionary containing the fan sensor information.
        :raises: IloConnectionError if failed connecting to the iLO.
        :ra
~~~

#### train_python-20 — `ce71a8d32b753176`
- Language: python; split: train
- Lengths: docstring 224 chars / 43 tokens; code 934 chars / 192 tokens
- Machine indicators: function names ['identifies', 'send_event_to_salt']; names in docstring ['identifies']; token overlap 0.592; comment ratio 0.19; repeated-line ratio 0.053; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
This function identifies whether the engine is running on the master
        or the minion and sends the data to the master event bus accordingly.

        :param result: It's a dictionary which has the final data and topic.
~~~
- Code excerpt:
~~~text
def send_event_to_salt(self, result):
        '''
        This function identifies whether the engine is running on the master
        or the minion and sends the data to the master event bus accordingly.

        :param result: It's a dictionary which has the final data and topic.

        '''
        if result['send'
~~~

#### train_python-21 — `bcea244d451d7261`
- Language: python; split: train
- Lengths: docstring 1192 chars / 301 tokens; code 2263 chars / 692 tokens
- Machine indicators: function names ['optimal_davies_pvalue']; names in docstring []; token overlap 0.708; comment ratio 0.0; repeated-line ratio 0.032; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
r"""Joint significance of statistics derived from chi2-squared distributions.

    Parameters
    ----------
    q : array_like
        Most significant of the independent test statistics, qₘᵢₙ(𝜌ᵥ) [1].
    mu : float
        Mean of the li
~~~
- Code excerpt:
~~~text
def optimal_davies_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid, pmin=None):
    r"""Joint significance of statistics derived from chi2-squared distributions.

    Parameters
    ----------
    q : array_like
        Most significant of the independent test statistics, qₘᵢₙ(𝜌ᵥ) [1].
    mu : float
        Mean
~~~

#### train_python-22 — `76f6f7352881b113`
- Language: python; split: train
- Lengths: docstring 169 chars / 39 tokens; code 922 chars / 251 tokens
- Machine indicators: function names ['PrintResponse']; names in docstring []; token overlap 0.31; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Prints the BatchJobService response.

  Args:
    batch_job_helper: a BatchJobHelper instance.
    response_xml: a string containing a response from the BatchJobService.
~~~
- Code excerpt:
~~~text
def PrintResponse(batch_job_helper, response_xml):
  """Prints the BatchJobService response.

  Args:
    batch_job_helper: a BatchJobHelper instance.
    response_xml: a string containing a response from the BatchJobService.
  """
  response = batch_job_helper.ParseResponse(response_xml)

  if 'rval' in response['muta
~~~

#### train_python-23 — `4c2c5712cfa6d2a2`
- Language: python; split: train
- Lengths: docstring 881 chars / 193 tokens; code 2336 chars / 603 tokens
- Machine indicators: function names ['plot_sfs']; names in docstring []; token overlap 0.636; comment ratio 0.083; repeated-line ratio 0.096; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Plot a site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum.
    yscale : string, optional
        Y axis scale.
    bins : int or array_like, int, optional

~~~
- Code excerpt:
~~~text
def plot_sfs(s, yscale='log', bins=None, n=None,
             clip_endpoints=True, label=None, plot_kwargs=None,
             ax=None):
    """Plot a site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum.
    yscale : string, optional
   
~~~

#### train_python-24 — `a741d153d4b85f30`
- Language: python; split: train
- Lengths: docstring 600 chars / 147 tokens; code 1183 chars / 309 tokens
- Machine indicators: function names ['packints_decode']; names in docstring ['packints_decode']; token overlap 0.723; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Decompress byte string to array of integers.

    This implementation only handles itemsizes 1, 8, 16, 32, and 64 bits.
    Install the imagecodecs package for decoding other integer sizes.

    Parameters
    ----------
    data : byte str
~~~
- Code excerpt:
~~~text
def packints_decode(data, dtype, numbits, runlen=0, out=None):
    """Decompress byte string to array of integers.

    This implementation only handles itemsizes 1, 8, 16, 32, and 64 bits.
    Install the imagecodecs package for decoding other integer sizes.

    Parameters
    ----------
    data : byte str
        D
~~~

#### train_python-25 — `4273dc2847ebd8ad`
- Language: python; split: train
- Lengths: docstring 2900 chars / 1074 tokens; code 4062 chars / 1545 tokens
- Machine indicators: function names ['Lehrer']; names in docstring ['Lehrer']; token overlap 0.921; comment ratio 0.0; repeated-line ratio 0.087; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        Nu_{S,L} = \left[\frac{0.03Re_S^{0.75}Pr}{1 + \frac{1.74(Pr-1)}
        {Re_S^{0.125}}}\right]\
~~~
- Code excerpt:
~~~text
def Lehrer(m, Dtank, Djacket, H, Dinlet, rho, Cp, k, mu, muw=None,
           isobaric_expansion=None, dT=None, inlettype='tangential',
           inletlocation='auto'):
    r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        N
~~~

#### train_python-26 — `d09639acd5a509b0`
- Language: python; split: train
- Lengths: docstring 1487 chars / 301 tokens; code 3350 chars / 856 tokens
- Machine indicators: function names ['get_correction']; names in docstring []; token overlap 0.625; comment ratio 0.0; repeated-line ratio 0.058; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Gets the Freysoldt correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute Freysoldt correction on.
                Requires following parameters in the DefectEntry to exist:

                   
~~~
- Code excerpt:
~~~text
def get_correction(self, entry):
        """
        Gets the Freysoldt correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute Freysoldt correction on.
                Requires following parameters in the DefectEntry to exist:

                    axis_grid (3 x NGX where N
~~~

#### train_python-27 — `4d8967c0d884c2c3`
- Language: python; split: train
- Lengths: docstring 330 chars / 73 tokens; code 4454 chars / 1251 tokens
- Machine indicators: function names ['fit', 'using']; names in docstring []; token overlap 0.15; comment ratio 0.164; repeated-line ratio 0.113; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Train a network with the quasi-Newton method.
        
        Args:
            X (np.array of float): feature matrix for training
            y (np.array of float): target values for training
            X_val (np.array of float): feature
~~~
- Code excerpt:
~~~text
def fit(self, X, y, X_val=None, y_val=None):
        """Train a network with the quasi-Newton method.
        
        Args:
            X (np.array of float): feature matrix for training
            y (np.array of float): target values for training
            X_val (np.array of float): feature matrix for validation
 
~~~

#### train_python-28 — `3b0815efb19a3a9d`
- Language: python; split: train
- Lengths: docstring 454 chars / 105 tokens; code 2207 chars / 576 tokens
- Machine indicators: function names ['read_adjacency_matrix']; names in docstring []; token overlap 0.38; comment ratio 0.101; repeated-line ratio 0.069; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Reads an edge list in csv format and returns the adjacency matrix in SciPy Sparse COOrdinate format.

    Inputs:  - file_path: The path where the adjacency matrix is stored.
             - separator: The delimiter among values (e.g. ",", "
~~~
- Code excerpt:
~~~text
def read_adjacency_matrix(file_path, separator, numbering="matlab"):
    """
    Reads an edge list in csv format and returns the adjacency matrix in SciPy Sparse COOrdinate format.

    Inputs:  - file_path: The path where the adjacency matrix is stored.
             - separator: The delimiter among values (e.g. ",", 
~~~

#### train_python-29 — `dfc70e7dcb44a052`
- Language: python; split: train
- Lengths: docstring 58 chars / 15 tokens; code 2120 chars / 437 tokens
- Machine indicators: function names ['add_empty_etd_ms_fields']; names in docstring []; token overlap 0.196; comment ratio 0.152; repeated-line ratio 0.304; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Add empty values for ETD_MS fields that don't have values.
~~~
- Code excerpt:
~~~text
def add_empty_etd_ms_fields(etd_ms_dict):
    """Add empty values for ETD_MS fields that don't have values."""
    # Determine which ETD MS elements are missing from the etd_ms_dict.
    for element in ETD_MS_ORDER:
        if element not in etd_ms_dict:
            # Try to create an element with content and qualifier
~~~

#### train_python-30 — `c804f902ae5474de`
- Language: python; split: train
- Lengths: docstring 271 chars / 53 tokens; code 1291 chars / 319 tokens
- Machine indicators: function names ['get_files_from_dir']; names in docstring []; token overlap 0.558; comment ratio 0.0; repeated-line ratio 0.037; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Recursively walk through a directory and get all python files and then walk
        through any potential directories that are found off current directory,
        so long as not within self.IGNORE_FOLDERS
        :return: all python files 
~~~
- Code excerpt:
~~~text
def get_files_from_dir(self, current_dir):
        """
        Recursively walk through a directory and get all python files and then walk
        through any potential directories that are found off current directory,
        so long as not within self.IGNORE_FOLDERS
        :return: all python files that were found o
~~~

### 3.1 JavaScript training samples

#### train_javascript-01 — `75810e5a4c309430`
- Language: javascript; split: train
- Lengths: docstring 172 chars / 39 tokens; code 138 chars / 39 tokens
- Machine indicators: function names ['GameMsgGenerator']; names in docstring ['GameMsgGenerator']; token overlap 0.08; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
## GameMsgGenerator constructor

Creates a new instance of GameMsgGenerator

@param {GameServer} gameServer Reference to the Game Server where
the generator is instantiated
~~~
- Code excerpt:
~~~text
function GameMsgGenerator(gameServer) {
    this.session = gameServer.session;
    this.sender = gameServer.name;
    this.stage = null;
}
~~~

#### train_javascript-02 — `81d1ceeac0596b85`
- Language: javascript; split: train
- Lengths: docstring 292 chars / 98 tokens; code 263 chars / 64 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.259; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Create a thumbnailed version of the image

@param  {Object} [options]
@param  {Number} [options.width=50]     Width of the thumbnail
@param  {Number} [options.height=50]    Height of the thumbnail
@param  {String} [options.fit=outbound] Fit
~~~
- Code excerpt:
~~~text
function(options) {
        options = options || {};

        return this.append([
            'thumbnail:width=' + (options.width || 50),
            'height=' + (options.height || 50),
            'fit=' + (options.fit || 'outbound')
        ].join(','));
    }
~~~

#### train_javascript-03 — `086c11f72364e216`
- Language: javascript; split: train
- Lengths: docstring 187 chars / 58 tokens; code 172 chars / 51 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.25; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Special-cased "all" method, applying the given route `path`,
middleware, and callback.

@name .all
@param {String} `path`
@param {Function} `callback`
@return {Object} `this` for chaining
~~~
- Code excerpt:
~~~text
function(path/*, callback*/) {
    this.lazyRouter();
    var route = this.router.route(path);
    route.all.apply(route, [].slice.call(arguments, 1));
    return this;
  }
~~~

#### train_javascript-04 — `658fee71b8a5ed32`
- Language: javascript; split: train
- Lengths: docstring 106 chars / 26 tokens; code 285 chars / 82 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.083; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes setting element opacity; code implements opacity handling.
- Docstring excerpt:
~~~text
Set the opacity of an element
@param {HTMLElement} element
@param {Number} opacity must be between 0 and 1
~~~
- Code excerpt:
~~~text
function (element, opacity) {
            var browser = ariaCoreBrowser;
            var isIE8OrLess = (browser.isIE8 || browser.isIE7);
            this.setOpacity = isIE8OrLess ? this._setOpacityLegacyIE : this._setOpacityW3C;
            this.setOpacity(element, opacity);
        }
~~~

#### train_javascript-05 — `f55e047ce14b006a`
- Language: javascript; split: train
- Lengths: docstring 161 chars / 46 tokens; code 259 chars / 72 tokens
- Machine indicators: function names ['handleReply']; names in docstring []; token overlap 0.148; comment ratio 0.0; repeated-line ratio 0.125; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Modifies the data to be in reply if in reply

@param {inboundMessage} inboundMessage
@param {Object} data - the data for the SparkPost API
@returns {Object} data
~~~
- Code excerpt:
~~~text
function handleReply(inboundMessage, data) {
  if (data.reply) {
    _.merge(data.content.headers, Transport.getReplyHeaders(inboundMessage));

    data.content.subject = Transport.getReplySubject(inboundMessage);

    delete data.reply;
  }

  return data;
}
~~~

#### train_javascript-06 — `3f907173462e3ff1`
- Language: javascript; split: train
- Lengths: docstring 126 chars / 37 tokens; code 136 chars / 46 tokens
- Machine indicators: function names ['copy']; names in docstring ['copy']; token overlap 0.2; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
[copy description]
@param  {[type]} from [description]
@param  {[type]} to   [description]
@return {[type]}      [description]
~~~
- Code excerpt:
~~~text
function copy(from, to){
  to = to || from;
  from = path.join(tpl, from);
  fs.createReadStream(from).pipe(fs.createWriteStream(to));
}
~~~

#### train_javascript-07 — `0e40d63ae5dc93ab`
- Language: javascript; split: train
- Lengths: docstring 39 chars / 9 tokens; code 401 chars / 81 tokens
- Machine indicators: function names ['clearTimeouts']; names in docstring []; token overlap 0.042; comment ratio 0.071; repeated-line ratio 0.167; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Clears queue and socket timeouts if any
~~~
- Code excerpt:
~~~text
function clearTimeouts() {
        // stop tracking time for network operations
        self._networkTime = contimer.stop(self._timerCtx, self.buildTimerId('network'));

        if (socketTimeout) {
            clearTimeout(socketTimeout);
            socketTimeout = null;
        }

        if (queueTimeout) {
       
~~~

#### train_javascript-08 — `df558ca2df493080`
- Language: javascript; split: train
- Lengths: docstring 28 chars / 7 tokens; code 428 chars / 86 tokens
- Machine indicators: function names ['getIndex']; names in docstring []; token overlap 0.043; comment ratio 0.0; repeated-line ratio 0.273; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Used to get real index name.
~~~
- Code excerpt:
~~~text
function getIndex(index) {
        if (this[_privateKey]._options.keysIgnoreCase && typeof index === 'string') {
            const indexLowerCase = index.toLowerCase();
            for (const key in this[_privateKey]._schema) {
                if (typeof key === 'string' && key.toLowerCase() === indexLowerCase) {

~~~

#### train_javascript-09 — `c0040d9a0e400039`
- Language: javascript; split: train
- Lengths: docstring 64 chars / 28 tokens; code 110 chars / 35 tokens
- Machine indicators: function names ['jsonToMm']; names in docstring ['jsonToMm']; token overlap 0.067; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Usage: jsonToMm( mmToJson( '../../../myGTD/myGTD.mm', false ) );
~~~
- Code excerpt:
~~~text
function jsonToMm(data) {
    startMap();
    traverseTree( data.map.node, startTag, endTag );
    endMap();
}
~~~

#### train_javascript-10 — `6023458e39663e01`
- Language: javascript; split: train
- Lengths: docstring 40 chars / 5 tokens; code 324 chars / 88 tokens
- Machine indicators: function names ['FSBackend']; names in docstring []; token overlap 0.053; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — constructor docstring matches the filesystem cache backend constructor.
- Docstring excerpt:
~~~text
Constructor for filesystem cache backend
~~~
- Code excerpt:
~~~text
function FSBackend(loadParameter) {
    this.loaded = false;
    this.index = [];
    this.location = typeof loadParameter === "string" && loadParameter.length > 0 ? loadParameter : process.cwd() + "/cache/";
    this.location = this.location.substr(this.location.length - 1) === "/" ? this.location : this.location + "/
~~~

#### train_javascript-11 — `f4a6fdceb979c675`
- Language: javascript; split: train
- Lengths: docstring 176 chars / 45 tokens; code 561 chars / 163 tokens
- Machine indicators: function names ['getProjectName']; names in docstring []; token overlap 0.119; comment ratio 0.133; repeated-line ratio 0.077; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Get name of the current project.

@param {Object} ctx - cordova context instance
@param {String} projectRoot - current root of the project

@return {String} name of the project
~~~
- Code excerpt:
~~~text
function getProjectName(ctx, projectRoot) {
  var cordova_util = ctx.requireCordovaModule('cordova-lib/src/cordova/util');
  var xml = cordova_util.projectConfig(projectRoot);
  var ConfigParser;

  // If we are running Cordova 5.4 or abova - use parser from cordova-common.
  // Otherwise - from cordova-lib.
  try {
  
~~~

#### train_javascript-12 — `fb63b0009830d941`
- Language: javascript; split: train
- Lengths: docstring 34 chars / 8 tokens; code 519 chars / 148 tokens
- Machine indicators: function names ['bundle', 'watchifyFile']; names in docstring []; token overlap 0.029; comment ratio 0.0; repeated-line ratio 0.056; bundle []; minified False
- Assessment: Strong alignment — docstring and function both describe compiling/watchifying a source file.
- Docstring excerpt:
~~~text
Compile and watchify sourced file.
~~~
- Code excerpt:
~~~text
function watchifyFile(src, out) {
    var opts = assign({}, watchify.args, {
        entries: src,
        standalone: "NavMesh",
        debug: true
    });
    var b = watchify(browserify(opts));
    function bundle() {
        return b.bundle()
            .on('error', gutil.log.bind(gutil, "Browserify Error"))
    
~~~

#### train_javascript-13 — `ca5186d717a13cae`
- Language: javascript; split: train
- Lengths: docstring 205 chars / 52 tokens; code 588 chars / 163 tokens
- Machine indicators: function names ['start']; names in docstring []; token overlap 0.091; comment ratio 0.0; repeated-line ratio 0.125; bundle []; minified False
- Assessment: Partial alignment — docstring describes a status wrapper; code begins implementing it.
- Docstring excerpt:
~~~text
This object is our generic status wrapper that lets us return information
about both WiFi and Native.

@public
@typedef {Object} combinedResult
@property {?Error} wifiResult
@property {?Error} nativeResult
~~~
- Code excerpt:
~~~text
function start (router, pskIdToSecret, networkType) {
  if (thaliMobileStates.started === true) {
    return Promise.reject(new Error('Call Stop!'));
  }
  thaliMobileStates.started = true;
  thaliMobileStates.networkType =
    networkType || global.NETWORK_TYPE || thaliMobileStates.networkType;

  return getWifiOrNati
~~~

#### train_javascript-14 — `2ead5044608a05a2`
- Language: javascript; split: train
- Lengths: docstring 36 chars / 5 tokens; code 733 chars / 153 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.0; comment ratio 0.0; repeated-line ratio 0.067; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is a vague phrase; code builds a validation message.
- Docstring excerpt:
~~~text
create custom message for validation
~~~
- Code excerpt:
~~~text
function (config, args) {
                var msg = config["Msg"];

                var format = config["Format"];
                if (format != undefined) {
                    _.extend(args, {
                        FormatedFrom: moment(args.From).format(format),
                        FormatedTo: moment(args.To).f
~~~

#### train_javascript-15 — `5c02f83f54cbe3c9`
- Language: javascript; split: train
- Lengths: docstring 43 chars / 10 tokens; code 455 chars / 112 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.0; comment ratio 0.0; repeated-line ratio 0.2; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is a vague event note; code handles suggestion clicks.
- Docstring excerpt:
~~~text
Listen for click event on suggestions list:
~~~
- Code excerpt:
~~~text
function (e) {
        var that = this,
            $el = $(e.target),
            index;

        if (!that.dropdownDisabled) {
            that.cancelFocus = true;
            that.el.focus();

            while ($el.length && !(index = $el.attr('data-index'))) {
                $el = $el.closest('.' + that.classes.s
~~~

#### train_javascript-16 — `4025ee8391560c6f`
- Language: javascript; split: train
- Lengths: docstring 58 chars / 21 tokens; code 289 chars / 95 tokens
- Machine indicators: function names ['uniq']; names in docstring ['uniq']; token overlap 0.095; comment ratio 0.0; repeated-line ratio 0.167; bundle []; minified False
- Assessment: Strong alignment — docstring and function both describe de-duplicating keywords.
- Docstring excerpt:
~~~text
Uniqueify keywords.

@param  {Array} `arr`
@return {Array}
~~~
- Code excerpt:
~~~text
function uniq(arr) {
  if (arr == null || arr.length === 0) {
    return [];
  }

  return _.reduce(arr, function(acc, ele) {
    var letter = exclusions.singleLetters;

    if (acc.indexOf(ele) === -1 && letter.indexOf(ele) === -1) {
      acc.push(ele);
    }
    return acc;
  }, []);
}
~~~

#### train_javascript-17 — `63e785d5dcf0713a`
- Language: javascript; split: train
- Lengths: docstring 73 chars / 15 tokens; code 501 chars / 102 tokens
- Machine indicators: function names ['takeGlobalQueue']; names in docstring []; token overlap 0.128; comment ratio 0.364; repeated-line ratio 0.091; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
Internal method to transfer globalQueue items to this context's
defQueue.
~~~
- Code excerpt:
~~~text
function takeGlobalQueue() {
            //Push all the globalDefQueue items into the context's defQueue
            if (globalDefQueue.length) {
                //Array splice in the values since the context code has a
                //local var ref to defQueue, so cannot just reassign the one
                //on co
~~~

#### train_javascript-18 — `83a855d99a004625`
- Language: javascript; split: train
- Lengths: docstring 113 chars / 64 tokens; code 392 chars / 104 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.148; comment ratio 0.071; repeated-line ratio 0.071; bundle []; minified False
- Assessment: Partial alignment — docstring gives a purpose but shares limited vocabulary with the code.
- Docstring excerpt:
~~~text
/*
单个接口出错处理
@param  {String} _bid   请求标识
@param  {String} _cid   接口标识
@param  {Object} _error 错误信息
@return {Void}
~~~
- Code excerpt:
~~~text
function(_bid,_cid,_error){
        var _batch = _xcache[_bid];
        if (!_batch||_doFilter(_error)) return;
        var _handler = _batch.h[_cid];
        if (!_handler) return;
        try{
            (_handler.e||f)(_error);
        }catch(ex){
            // ignore
            if (DEBUG) throw ex;
            c
~~~

#### train_javascript-19 — `9216a4442777611b`
- Language: javascript; split: train
- Lengths: docstring 294 chars / 78 tokens; code 633 chars / 149 tokens
- Machine indicators: function names ['clearDefaultValue', 'default']; names in docstring ['clearDefaultValue', 'default']; token overlap 0.123; comment ratio 0.067; repeated-line ratio 0.133; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Clears SQL type escapes (Two quote '' to one quote ') and strips beginning and trailing quotes around string.
@private
@param {string} string - Default
@returns {string|boolean|undefined}
@example
var clear = clearDefaultValue("'No ''value'
~~~
- Code excerpt:
~~~text
function clearDefaultValue(string) {
    // Does not support SQL functions. IMHO it is better to handle sql function default values in RDMS.
    var defaultValue = undefined, lowercaseString, booleanValue;
    if (string.charAt(0) === "'" || string.charAt(0) === '"') {
        string = string.substring(1, string.length
~~~

#### train_javascript-20 — `09435432c4db5e2a`
- Language: javascript; split: train
- Lengths: docstring 111 chars / 33 tokens; code 257 chars / 108 tokens
- Machine indicators: function names ['parse_field_top']; names in docstring []; token overlap 0.167; comment ratio 0.0; repeated-line ratio 0.111; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring contains debug logs, not a specification.
- Docstring excerpt:
~~~text
debug.log('obj = ', obj); debug.log('key = ', key); debug.log('value = ', value); /* Parse full top level field
~~~
- Code excerpt:
~~~text
function parse_field_top(obj, key, value) {
			if( is.array(obj['$'+key]) ) {
				obj['$'+key] = obj['$'+key].concat(value);
			} else if( is.obj(obj['$'+key]) ) {
				obj['$'+key] = merge(obj['$'+key], value);
			} else {
				obj['$'+key] = value;
			}
		}
~~~

#### train_javascript-21 — `25769d4927cf34f4`
- Language: javascript; split: train
- Lengths: docstring 293 chars / 84 tokens; code 891 chars / 223 tokens
- Machine indicators: function names ['createUrlTree']; names in docstring []; token overlap 0.103; comment ratio 0.0; repeated-line ratio 0.143; bundle ['bundler/minified marker']; minified False
- Assessment: Mismatched/unusable — docstring is an Angular license header, not a function description.
- Docstring excerpt:
~~~text
@license
Copyright Google Inc. All Rights Reserved.

Use of this source code is governed by an MIT-style license that can be
found in the LICENSE file at https://angular.io/license
 
@param {?} route
@param {?} urlTree
@param {?} commands
@
~~~
- Code excerpt:
~~~text
function createUrlTree(route, urlTree, commands, queryParams, fragment) {
    if (commands.length === 0) {
        return tree(urlTree.root, urlTree.root, urlTree, queryParams, fragment);
    }
    var /** @type {?} */ nav = computeNavigation(commands);
    if (nav.toRoot()) {
        return tree(urlTree.root, new __WE
~~~

#### train_javascript-22 — `f501546d3685fbe8`
- Language: javascript; split: train
- Lengths: docstring 137 chars / 48 tokens; code 608 chars / 255 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.042; comment ratio 0.069; repeated-line ratio 0.12; bundle []; minified False
- Assessment: Partial alignment — docstring cites a theorem; code implements collision detection.
- Docstring excerpt:
~~~text
Separating Axis Theorem https://gamedevelopment.tutsplus.com/tutorials/collision-detection-using-the-separating-axis-theorem--gamedev-169
~~~
- Code excerpt:
~~~text
function(other) {
		var r0 = this._points();
		var r1 = other._points();
		var axes = [
			toAxis(r0[0], r0[1]),
			toAxis(r0[0], r0[3])
		];
		var i, pr0, pr1;

		if (this._rotation !== other._rotation) {
			// Only separate with r1 axis if the rotation is different,
			// else it's enough to separate r0 and r1 with r
~~~

#### train_javascript-23 — `ec756e323ffd0748`
- Language: javascript; split: train
- Lengths: docstring 341 chars / 85 tokens; code 12458 chars / 2757 tokens
- Machine indicators: function names ['expandRng', 'findBlockEndPoint', 'findLeaf', 'findParentContainer', 'findParentContentEditable', 'findSelectorEndPoint', 'findSpace', 'findWordEndPoint', 'isBogusBr', 'walks']; names in docstring []; token overlap 0.097; comment ratio 0.109; repeated-line ratio 0.299; bundle ['many functions/braces', 'very long file']; minified False
- Assessment: Strong alignment — docstring and code both describe expanding a range-like object.
- Docstring excerpt:
~~~text
Expands the specified range like object to depending on format.

For example on block formats it will move the start/end position
to the beginning of the current block.

@private
@param {Object} rng Range like object.
@param {Array} format 
~~~
- Code excerpt:
~~~text
function expandRng(rng, format, remove) {
        var lastIdx, leaf, endPoint,
          startContainer = rng.startContainer,
          startOffset = rng.startOffset,
          endContainer = rng.endContainer,
          endOffset = rng.endOffset;

        // This function walks up the tree if there is no siblings befor
~~~

#### train_javascript-24 — `645a6144685d71bd`
- Language: javascript; split: train
- Lengths: docstring 50 chars / 10 tokens; code 5688 chars / 1396 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.012; comment ratio 0.244; repeated-line ratio 0.155; bundle []; minified False
- Assessment: Partial alignment — short constructor note; code is a long initialiser.
- Docstring excerpt:
~~~text
/*
Initialize the plugin
Called by the constructor
~~~
- Code excerpt:
~~~text
function () {
            var self = this;

            // Init check for class existence
            if (self.options.map.cssClass === "" || $("." + self.options.map.cssClass, self.container).length === 0) {
                throw new Error("The map class `" + self.options.map.cssClass + "` doesn't exists");
          
~~~

#### train_javascript-25 — `4a5825ba38db1479`
- Language: javascript; split: train
- Lengths: docstring 221 chars / 55 tokens; code 903 chars / 239 tokens
- Machine indicators: function names ['_resolveTouchInputConflicts']; names in docstring ['_resolveTouchInputConflicts']; token overlap 0.167; comment ratio 0.0; repeated-line ratio 0.345; bundle []; minified False
- Assessment: Strong alignment — docstring and function both describe resolving touch conflicts.
- Docstring excerpt:
~~~text
Resolves conflicting active tools when activating a tool for touch interaction
@private
@function _resolveTouchInputConflicts

@param {Object} tool
@param {HTMLElement} element
@param {Object} options
@returns {undefined}
~~~
- Code excerpt:
~~~text
function _resolveTouchInputConflicts(tool, element, options) {
  const activeTouchTool = store.state.tools.find(
    t =>
      t.element === element &&
      t.mode === 'active' &&
      t.options.isTouchActive === true
  );

  const activeMultiTouchToolWithOneTouchPointer = store.state.tools.find(
    t =>
      t.el
~~~

#### train_javascript-26 — `a3c1ab76bbd99b4e`
- Language: javascript; split: train
- Lengths: docstring 142 chars / 32 tokens; code 1203 chars / 512 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.074; comment ratio 0.0; repeated-line ratio 0.051; bundle []; minified False
- Assessment: Strong alignment — docstring describes date formatting; code formats dates.
- Docstring excerpt:
~~~text
Returns a date string formatted according to the given format string

@param  String  format string
@param  Object  Date object
@return String
~~~
- Code excerpt:
~~~text
function(format, date) {
		var self = this,
			output, d, dw, m, y, h, g, i, s;
		
		if (! date) {
			date = new Date();
		}
		
		h  = date[self.getHours]();
		g  = h > 12 ? h - 12 : h;
		i  = date[self.getMinutes]();
		s  = date[self.getSeconds]();
		d  = date[self.getDate]();
		dw = date[self.getDay]();
		m  = date[s
~~~

#### train_javascript-27 — `ad1f6ca2b26d4ba6`
- Language: javascript; split: train
- Lengths: docstring 6 chars / 3 tokens; code 7244 chars / 1990 tokens
- Machine indicators: function names ['Sole', '_classCallCheck', '_createClass', 'compareScoreCards', 'defineProperties', 'escapeRegExp', 'getResultWeights', 'isItemAboveMinimum', 'isThereAnythingRelatedTo', 'scoreThis', 'startsWith']; names in docstring []; token overlap 0.0; comment ratio 0.991; repeated-line ratio 0.292; bundle ['bundler/minified marker', 'many functions/braces']; minified True
- Assessment: Strong alignment — docstring describes clearing a default value; code does so.
- Docstring excerpt:
~~~text
/* 197
~~~
- Code excerpt:
~~~text
function(module,exports,__webpack_require__){"use strict";Object.defineProperty(exports,"__esModule",{value:true});var _createClass=function(){function defineProperties(target,props){for(var i=0;i<props.length;i++){var descriptor=props[i];descriptor.enumerable=descriptor.enumerable||false;descriptor.configurable=true;i
~~~

#### train_javascript-28 — `d1148fd1d7a536c7`
- Language: javascript; split: train
- Lengths: docstring 150 chars / 27 tokens; code 2371 chars / 336 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.0; comment ratio 0.054; repeated-line ratio 0.314; bundle []; minified False
- Assessment: Mismatched/unusable — docstring is a section banner; code handles messages.
- Docstring excerpt:
~~~text
*****************************************************
P R I V A T E   M E T H O D S            *
*****************************************************
~~~
- Code excerpt:
~~~text
function(messageContent, addFromService) {
                var threadData = formatDataToMakeConversation(messageContent);

                if (addFromService) {
                    fireEvent('threadEvents', {
                        type: 'THREAD_NEW',
                        result: {
                            threa
~~~

#### train_javascript-29 — `a2f7ff4711090a44`
- Language: javascript; split: train
- Lengths: docstring 170 chars / 51 tokens; code 889 chars / 268 tokens
- Machine indicators: function names ['getRectangleFromImagePlane']; names in docstring ['getRectangleFromImagePlane']; token overlap 0.125; comment ratio 0.049; repeated-line ratio 0.41; bundle []; minified False
- Assessment: Partial alignment — docstring describes a rectangle helper; code computes it.
- Docstring excerpt:
~~~text
Returns a rectangle from the imagePlane.
@export @public @method
@name getRectangleFromImagePlane

@param  {Object} imagePlane The imagePlane.
@returns {Object} The rect.
~~~
- Code excerpt:
~~~text
function getRectangleFromImagePlane(imagePlane) {
  // Get the points
  const topLeft = imagePointToPatientPoint(
    {
      x: 0,
      y: 0,
    },
    imagePlane
  );
  const topRight = imagePointToPatientPoint(
    {
      x: imagePlane.columns,
      y: 0,
    },
    imagePlane
  );
  const bottomLeft = imagePoin
~~~

#### train_javascript-30 — `ab337469dd8f33d1`
- Language: javascript; split: train
- Lengths: docstring 135 chars / 11 tokens; code 1957 chars / 489 tokens
- Machine indicators: function names ['handleResponse', 'onFileReady']; names in docstring []; token overlap 0.037; comment ratio 0.101; repeated-line ratio 0.186; bundle []; minified False
- Assessment: Mismatched/unusable — docstring is a banner, not a description of file processing.
- Docstring excerpt:
~~~text
====================================================== Process the file content. ======================================================
~~~
- Code excerpt:
~~~text
function onFileReady(err, data) {
    if (err) {
      if (onDone) onDone(err)
      else console.error('>>>' + err);
      return;
    }

    var fileContent = data.toString();

    var msg = {
      text:    fileContent,
      mode:    flavor,
      context: context
    };

    var jsonMsg = JSON.stringify(msg);

   
~~~

### 3.1 Python validation samples

#### valid_python-01 — `dc5a97ab0f3262e3`
- Language: python; split: valid
- Lengths: docstring 164 chars / 42 tokens; code 390 chars / 99 tokens
- Machine indicators: function names ['new_account']; names in docstring []; token overlap 0.581; comment ratio 0.0; repeated-line ratio 0.1; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Creates new account, appends it to the :class:`Wallet`'s account list and returns it.

        :param label: account label as `str`
        :rtype: :class:`Account`
~~~
- Code excerpt:
~~~text
def new_account(self, label=None):
        """
        Creates new account, appends it to the :class:`Wallet`'s account list and returns it.

        :param label: account label as `str`
        :rtype: :class:`Account`
        """
        acc, addr = self._backend.new_account(label=label)
        assert acc.index == l
~~~

#### valid_python-02 — `7ce0154f9961f2df`
- Language: python; split: valid
- Lengths: docstring 62 chars / 15 tokens; code 368 chars / 114 tokens
- Machine indicators: function names ['list_trilegal_filtersystems']; names in docstring []; token overlap 0.435; comment ratio 0.0; repeated-line ratio 0.125; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
This just lists all the filter systems available for TRILEGAL.
~~~
- Code excerpt:
~~~text
def list_trilegal_filtersystems():
    '''
    This just lists all the filter systems available for TRILEGAL.

    '''

    print('%-40s %s' % ('FILTER SYSTEM NAME','DESCRIPTION'))
    print('%-40s %s' % ('------------------','-----------'))
    for key in sorted(TRILEGAL_FILTER_SYSTEMS.keys()):
        print('%-40s %s
~~~

#### valid_python-03 — `044d0aabc779fd2e`
- Language: python; split: valid
- Lengths: docstring 175 chars / 33 tokens; code 552 chars / 117 tokens
- Machine indicators: function names ['list_workflow_outputs']; names in docstring []; token overlap 0.556; comment ratio 0.0; repeated-line ratio 0.071; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Get a list of outputs from the workflow that are saved to S3. To get resolved locations call workflow status.
        Args:
            None

        Returns:
            list
~~~
- Code excerpt:
~~~text
def list_workflow_outputs(self):
        '''
        Get a list of outputs from the workflow that are saved to S3. To get resolved locations call workflow status.
        Args:
            None

        Returns:
            list
        '''
        workflow_outputs = []
        for task in self.tasks:
            for o
~~~

#### valid_python-04 — `35d9d1bcbff191b5`
- Language: python; split: valid
- Lengths: docstring 56 chars / 24 tokens; code 176 chars / 61 tokens
- Machine indicators: function names ['loose_in']; names in docstring ['loose_in']; token overlap 0.467; comment ratio 0.0; repeated-line ratio 0.143; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
pl = ['bcd','xabcxx','x','y']
        loose_in(pl,'abc')
~~~
- Code excerpt:
~~~text
def loose_in(pl,k):
    '''
        pl = ['bcd','xabcxx','x','y']
        loose_in(pl,'abc')
        
    '''
    cond = some(pl,lambda ele:(k in ele))['cond']
    return(cond)
~~~

#### valid_python-05 — `ad2696051f5223be`
- Language: python; split: valid
- Lengths: docstring 22 chars / 5 tokens; code 306 chars / 70 tokens
- Machine indicators: function names ['set_active_client']; names in docstring []; token overlap 0.235; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Set the active client.
~~~
- Code excerpt:
~~~text
async def set_active_client(self, set_active_client_request):
        """Set the active client."""
        response = hangouts_pb2.SetActiveClientResponse()
        await self._pb_request('clients/setactiveclient',
                               set_active_client_request, response)
        return response
~~~

#### valid_python-06 — `4c01de5fb46fc4d0`
- Language: python; split: valid
- Lengths: docstring 66 chars / 12 tokens; code 938 chars / 192 tokens
- Machine indicators: function names ['receive']; names in docstring ['receive']; token overlap 0.211; comment ratio 0.0; repeated-line ratio 0.19; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Receive ASGI websocket messages, ensuring valid state transitions.
~~~
- Code excerpt:
~~~text
async def receive(self) -> Message:
        """
        Receive ASGI websocket messages, ensuring valid state transitions.
        """
        if self.client_state == WebSocketState.CONNECTING:
            message = await self._receive()
            message_type = message["type"]
            assert message_type == "web
~~~

#### valid_python-07 — `15ee49c2de2dc6f7`
- Language: python; split: valid
- Lengths: docstring 427 chars / 110 tokens; code 593 chars / 161 tokens
- Machine indicators: function names ['iexSymbolsDF']; names in docstring []; token overlap 0.836; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
This call returns an array of symbols the Investors Exchange supports for trading.
    This list is updated daily as of 7:45 a.m. ET. Symbols may be added or removed by the Investors Exchange after the list was produced.

    https://iexclo
~~~
- Code excerpt:
~~~text
def iexSymbolsDF(token='', version=''):
    '''This call returns an array of symbols the Investors Exchange supports for trading.
    This list is updated daily as of 7:45 a.m. ET. Symbols may be added or removed by the Investors Exchange after the list was produced.

    https://iexcloud.io/docs/api/#iex-symbols
    8
~~~

#### valid_python-08 — `e38f42f44373a366`
- Language: python; split: valid
- Lengths: docstring 227 chars / 75 tokens; code 526 chars / 172 tokens
- Machine indicators: function names ['mapi']; names in docstring ['mapi']; token overlap 0.562; comment ratio 0.278; repeated-line ratio 0.056; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
#mapi     v不作为map_func参数,共享相同的f,共享相同的o
        #         NOT take value as a param for map_func
        #         share common other_args
        #         share common map_func
        #         common_func(index,*common_args)
~~~
- Code excerpt:
~~~text
def mapi(ol,map_func,map_func_args=[]):
    '''
        #mapi     v不作为map_func参数,共享相同的f,共享相同的o
        #         NOT take value as a param for map_func
        #         share common other_args
        #         share common map_func
        #         common_func(index,*common_args)
    '''
    lngth = ol.__len__()
   
~~~

#### valid_python-09 — `262aa7b1c6b8897a`
- Language: python; split: valid
- Lengths: docstring 120 chars / 32 tokens; code 488 chars / 129 tokens
- Machine indicators: function names ['get']; names in docstring ['get']; token overlap 0.394; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Get the global config values by keys.

    Example:

    \b
    ```bash
    $ polyaxon config get host http_port
    ```
~~~
- Code excerpt:
~~~text
def get(keys):
    """Get the global config values by keys.

    Example:

    \b
    ```bash
    $ polyaxon config get host http_port
    ```
    """
    _config = GlobalConfigManager.get_config_or_default()

    if not keys:
        return

    print_values = {}
    for key in keys:
        if hasattr(_config, key):

~~~

#### valid_python-10 — `0f7d9e806b45d75e`
- Language: python; split: valid
- Lengths: docstring 183 chars / 37 tokens; code 818 chars / 161 tokens
- Machine indicators: function names ['add_annotation_comment']; names in docstring []; token overlap 0.571; comment ratio 0.0; repeated-line ratio 0.118; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Sets the annotation comment. Raises CardinalityError if
        already set. OrderError if no annotator defined before.
        Raises SPDXValueError if comment is not free form text.
~~~
- Code excerpt:
~~~text
def add_annotation_comment(self, doc, comment):
        """Sets the annotation comment. Raises CardinalityError if
        already set. OrderError if no annotator defined before.
        Raises SPDXValueError if comment is not free form text.
        """
        if len(doc.annotations) != 0:
            if not self.ann
~~~

#### valid_python-11 — `7538f4823a57518a`
- Language: python; split: valid
- Lengths: docstring 1123 chars / 267 tokens; code 1284 chars / 310 tokens
- Machine indicators: function names ['spread']; names in docstring ['spread']; token overlap 0.973; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
This returns an array of effective spread, eligible volume, and price improvement of a stock, by market.
    Unlike volume-by-venue, this will only return a venue if effective spread is not ‘N/A’. Values are sorted in descending order by ef
~~~
- Code excerpt:
~~~text
def spread(symbol, token='', version=''):
    '''This returns an array of effective spread, eligible volume, and price improvement of a stock, by market.
    Unlike volume-by-venue, this will only return a venue if effective spread is not ‘N/A’. Values are sorted in descending order by effectiveSpread.
    Lower effect
~~~

#### valid_python-12 — `2050153d7099369e`
- Language: python; split: valid
- Lengths: docstring 640 chars / 169 tokens; code 5413 chars / 1648 tokens
- Machine indicators: function names ['plot_fit_individuals_lens_and_source_planes']; names in docstring []; token overlap 0.5; comment ratio 0.0; repeated-line ratio 0.267; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit betwe
~~~
- Code excerpt:
~~~text
def plot_fit_individuals_lens_and_source_planes(
        fit, should_plot_mask=True, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        should_plot_image_plane_pix=False,
        should_plot_image=False,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
    
~~~

#### valid_python-13 — `a3ad932b0f0495a9`
- Language: python; split: valid
- Lengths: docstring 512 chars / 108 tokens; code 1149 chars / 261 tokens
- Machine indicators: function names ['check_compatibility']; names in docstring []; token overlap 0.639; comment ratio 0.0; repeated-line ratio 0.069; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Parameters
        ----------
        one_img: str or img-like object.
            See NeuroImage constructor docstring.

        anoter_img: str or img-like object.
            See NeuroImage constructor docstring.
            If None will
~~~
- Code excerpt:
~~~text
def check_compatibility(self, one_img, another_img=None):
        """
        Parameters
        ----------
        one_img: str or img-like object.
            See NeuroImage constructor docstring.

        anoter_img: str or img-like object.
            See NeuroImage constructor docstring.
            If None will u
~~~

#### valid_python-14 — `5844a2ea54c7bb47`
- Language: python; split: valid
- Lengths: docstring 669 chars / 186 tokens; code 1135 chars / 301 tokens
- Machine indicators: function names ['stream_frames']; names in docstring ['stream_frames']; token overlap 0.727; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
Stream measured frames from QTM until :func:`~qtm.QRTConnection.stream_frames_stop`
           is called.


        :param frames: Which frames to receive, possible values are 'allframes',
            'frequency:n' or 'frequencydivisor:n' w
~~~
- Code excerpt:
~~~text
async def stream_frames(self, frames="allframes", components=None, on_packet=None):
        """Stream measured frames from QTM until :func:`~qtm.QRTConnection.stream_frames_stop`
           is called.


        :param frames: Which frames to receive, possible values are 'allframes',
            'frequency:n' or 'freque
~~~

#### valid_python-15 — `63a36cfdd9afacca`
- Language: python; split: valid
- Lengths: docstring 442 chars / 100 tokens; code 1188 chars / 331 tokens
- Machine indicators: function names ['Hfus', 'directly', 'handles', 'list_methods', 'of']; names in docstring ['directly', 'handles', 'of']; token overlap 0.566; comment ratio 0.032; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes the function and code implements it.
- Docstring excerpt:
~~~text
This function handles the calculation of a chemical's enthalpy of fusion.
    Generally this, is used by the chemical class, as all parameters are passed.
    Calling the function directly works okay.

    Enthalpy of fusion is a weak funct
~~~
- Code excerpt:
~~~text
def Hfus(T=298.15, P=101325, MW=None, AvailableMethods=False, Method=None, CASRN=''):  # pragma: no cover
    '''This function handles the calculation of a chemical's enthalpy of fusion.
    Generally this, is used by the chemical class, as all parameters are passed.
    Calling the function directly works okay.

    E
~~~

### 3.1 JavaScript validation samples

#### valid_javascript-01 — `e60a07913bf36c80`
- Language: javascript; split: valid
- Lengths: docstring 119 chars / 30 tokens; code 279 chars / 63 tokens
- Machine indicators: function names ['closest']; names in docstring ['closest']; token overlap 0.316; comment ratio 0.0; repeated-line ratio 0.222; bundle []; minified False
- Assessment: Strong alignment — docstring and code both find the closest matching parent.
- Docstring excerpt:
~~~text
Finds the closest parent that matches a selector.

@param {Element} element
@param {String} selector
@return {Function}
~~~
- Code excerpt:
~~~text
function closest (element, selector) {
    while (element && element.nodeType !== DOCUMENT_NODE_TYPE) {
        if (typeof element.matches === 'function' &&
            element.matches(selector)) {
          return element;
        }
        element = element.parentNode;
    }
}
~~~

#### valid_javascript-02 — `b4df2d5bec202438`
- Language: javascript; split: valid
- Lengths: docstring 20 chars / 7 tokens; code 222 chars / 78 tokens
- Machine indicators: function names ['stripZeros']; names in docstring []; token overlap 0.0; comment ratio 0.0; repeated-line ratio 0.111; bundle []; minified False
- Assessment: Mismatched/unusable — docstring is an import source, not a description.
- Docstring excerpt:
~~~text
from ethereumjs-util
~~~
- Code excerpt:
~~~text
function stripZeros(aInput) {
  var a = aInput; // eslint-disable-line
  var first = a[0]; // eslint-disable-line
  while (a.length > 0 && first.toString() === '0') {
    a = a.slice(1);
    first = a[0];
  }
  return a;
}
~~~

#### valid_javascript-03 — `3ccd0ff36848cee0`
- Language: javascript; split: valid
- Lengths: docstring 188 chars / 47 tokens; code 277 chars / 87 tokens
- Machine indicators: function names ['transObjectAttrIntoArray']; names in docstring []; token overlap 0.152; comment ratio 0.0; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Strong alignment — docstring describes sorting object attributes; code sorts them.
- Docstring excerpt:
~~~text
sort Object attributes by function
and transfer them into array
@param  {Object} obj Object form from numric
@param  {Function} fn sort function
@return {Array} the sorted attirbutes array
~~~
- Code excerpt:
~~~text
function transObjectAttrIntoArray(obj) {
  var fn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : function (a, b) {
    return +a - +b;
  };

  return _Object$keys(obj).sort(fn).reduce(function (order, key) {
    return order.concat(obj[key]);
  }, []);
}
~~~

#### valid_javascript-04 — `533e02ad6ccf6e24`
- Language: javascript; split: valid
- Lengths: docstring 30 chars / 7 tokens; code 342 chars / 61 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.067; comment ratio 0.0; repeated-line ratio 0.111; bundle []; minified False
- Assessment: Strong alignment — docstring describes wrapping an error; code wraps errors.
- Docstring excerpt:
~~~text
Wraps error into a SystemError
~~~
- Code excerpt:
~~~text
function(msg, code, errorStr, cb) {
                return function(error, data) {
                    if (error) {
                        error = json_rpc.newSysError(msg, code, errorStr,
                                                     error);
                    }
                    cb(error, data);
          
~~~

#### valid_javascript-05 — `575d018afba2f151`
- Language: javascript; split: valid
- Lengths: docstring 7 chars / 1 tokens; code 131 chars / 44 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.0; comment ratio 0.0; repeated-line ratio 0.143; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is only "private"; code is an error handler.
- Docstring excerpt:
~~~text
private
~~~
- Code excerpt:
~~~text
function (replacer, e) {
  var message = e;
  if (e instanceof Error) {
    message = e.message;
  }
  replacer.emit('error', e);
}
~~~

#### valid_javascript-06 — `18f4551ead195386`
- Language: javascript; split: valid
- Lengths: docstring 81 chars / 28 tokens; code 375 chars / 140 tokens
- Machine indicators: function names ['ZarinPal']; names in docstring ['ZarinPal']; token overlap 0.138; comment ratio 0.0; repeated-line ratio 0.1; bundle []; minified False
- Assessment: Strong alignment — constructor docstring matches the ZarinPal constructor.
- Docstring excerpt:
~~~text
Constructor for ZarinPal object.
@param {String} MerchantID
@param {bool} sandbox
~~~
- Code excerpt:
~~~text
function ZarinPal(MerchantID, sandbox) {
	if (typeof MerchantID === 'string' && MerchantID.length === config.merchantIDLength) {
		this.merchant = MerchantID;
	} else {
		console.error('The MerchantID must be ' + config.merchantIDLength + ' Characters.');
		return false;
	}
	this.sandbox = sandbox || false;

	this.url 
~~~

#### valid_javascript-07 — `804a3b533d299a4f`
- Language: javascript; split: valid
- Lengths: docstring 80 chars / 14 tokens; code 526 chars / 148 tokens
- Machine indicators: function names ['addPlugin']; names in docstring []; token overlap 0.129; comment ratio 0.0; repeated-line ratio 0.222; bundle []; minified False
- Assessment: Strong alignment — docstring describes adding a plugin; code adds a plugin.
- Docstring excerpt:
~~~text
Add plugin function. Plugin function must return array with log argument values.
~~~
- Code excerpt:
~~~text
function addPlugin(useLevel, fn) {
  let pluginFn = fn;
  if (typeof fn === 'undefined' && typeof useLevel === 'function') {
    pluginFn = useLevel;
  }

  if (typeof pluginFn !== 'function') {
    throw new Error('Plugin must be a function!');
  }

  if (typeof useLevel === 'string') {
    pluginFn = (id, level, stat
~~~

#### valid_javascript-08 — `b421b8324fd24a06`
- Language: javascript; split: valid
- Lengths: docstring 49 chars / 7 tokens; code 394 chars / 92 tokens
- Machine indicators: function names ['TransformStreamCloseReadable']; names in docstring []; token overlap 0.0; comment ratio 0.077; repeated-line ratio 0.2; bundle []; minified False
- Assessment: Partial alignment — generic controller description; code implements close-readable logic.
- Docstring excerpt:
~~~text
Methods on the transform stream controller object
~~~
- Code excerpt:
~~~text
function TransformStreamCloseReadable(transformStream) {
  // console.log('TransformStreamCloseReadable()');

  if (transformStream._errored === true) {
    throw new TypeError('TransformStream is already errored');
  }

  if (transformStream._readableClosed === true) {
    throw new TypeError('Readable side is already
~~~

#### valid_javascript-09 — `d248ecfa5228c1a8`
- Language: javascript; split: valid
- Lengths: docstring 5 chars / 1 tokens; code 456 chars / 99 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.0; comment ratio 0.077; repeated-line ratio 0.077; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is only "multi"; code handles a multi-select value.
- Docstring excerpt:
~~~text
multi
~~~
- Code excerpt:
~~~text
function (val) {
            var unique;
            if (this.select) {
                this.select.val(val);
            } else {
                unique = [];
                // filter out duplicates
                $(val).each(function () {
                    if (indexOf(this, unique) < 0) unique.push(this);
       
~~~

#### valid_javascript-10 — `2f7fe037dbcdb628`
- Language: javascript; split: valid
- Lengths: docstring 55 chars / 13 tokens; code 688 chars / 189 tokens
- Machine indicators: function names ['parsePropertyValue']; names in docstring []; token overlap 0.051; comment ratio 0.0; repeated-line ratio 0.053; bundle []; minified False
- Assessment: Strong alignment — docstring describes extracting a property value; code does so.
- Docstring excerpt:
~~~text
extracts the property value out of an itemprop DOM node
~~~
- Code excerpt:
~~~text
function parsePropertyValue (node) {
    var $node = $(node);

    if ($node.is('meta')) {
      return resolveAttribute($node, 'content');
    } else if ($node.is('audio,embed,iframe,img,source,track,video')) {
      return resolveUrlAttribute($node, 'src');
    } else if ($node.is('a,area,link')) {
      return resol
~~~

#### valid_javascript-11 — `31de0d8533c5c552`
- Language: javascript; split: valid
- Lengths: docstring 206 chars / 55 tokens; code 798 chars / 203 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.105; comment ratio 0.0; repeated-line ratio 0.13; bundle []; minified False
- Assessment: Strong alignment — docstring describes creating a file; code creates file metadata.
- Docstring excerpt:
~~~text
Create a new file.

@param {string} path - File path
@param body - File content
@param {string} contentType - File content-type
@returns {Promise} Resolves with the response of the network request

@private
~~~
- Code excerpt:
~~~text
function (path, body, contentType/*, options*/) {
    return this._getParentId(path).then((parentId) => {
      const fileName = baseName(path);
      const metadata = {
        title: metaTitleFromFileName(fileName),
        mimeType: contentType,
        parents: [{
          kind: "drive#fileLink",
          id: par
~~~

#### valid_javascript-12 — `9bcf47490bbb2abe`
- Language: javascript; split: valid
- Lengths: docstring 6 chars / 3 tokens; code 682 chars / 251 tokens
- Machine indicators: function names ['_toISOString', 'pad']; names in docstring []; token overlap 0.0; comment ratio 0.222; repeated-line ratio 0.071; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is a numeric marker, not a description.
- Docstring excerpt:
~~~text
/* 330
~~~
- Code excerpt:
~~~text
function(module, exports) {

	/**
	 * Polyfill from <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/toISOString>.
	 */
	var pad = function pad(n) {
	  return (n < 10 ? '0' : '') + n;
	};
	
	var _toISOString = typeof Date.prototype.toISOString === 'function' ? function _toISOString
~~~

#### valid_javascript-13 — `2c4540ceaf02f707`
- Language: javascript; split: valid
- Lengths: docstring 20 chars / 3 tokens; code 1649 chars / 379 tokens
- Machine indicators: function names []; names in docstring []; token overlap 0.009; comment ratio 0.286; repeated-line ratio 0.0; bundle []; minified False
- Assessment: Weak/unclear alignment — docstring is a section header; code renders a view.
- Docstring excerpt:
~~~text
Rendering ----------
~~~
- Code excerpt:
~~~text
function() {

	this.$el.empty();

        // A special markup can be given in the `properties.markup` property. This might be handy
        // if e.g. arrowhead markers should be `<image>` elements or any other element than `<path>`s.
        // `.connection`, `.connection-wrap`, `.marker-source` and `.marker-target` s
~~~

#### valid_javascript-14 — `33d94a8d3d599972`
- Language: javascript; split: valid
- Lengths: docstring 174 chars / 49 tokens; code 869 chars / 253 tokens
- Machine indicators: function names ['processStdin']; names in docstring []; token overlap 0.159; comment ratio 0.0; repeated-line ratio 0.172; bundle []; minified False
- Assessment: Strong alignment — docstring describes processing stdin; code reads and processes stdin.
- Docstring excerpt:
~~~text
Process all input from stdin.

@params opts {Object} Bunyan options object.
@param stylize {Function} Output stylize function to use.
@param callback {Function} `function ()`
~~~
- Code excerpt:
~~~text
function processStdin(opts, stylize, callback) {
    var leftover = '';  // Left-over partial line from last chunk.
    var stdin = process.stdin;
    stdin.resume();
    stdin.setEncoding('utf8');
    stdin.on('data', function (chunk) {
        var lines = chunk.split(/\r\n|\n/);
        var length = lines.length;
   
~~~

#### valid_javascript-15 — `34802d105f5d5e59`
- Language: javascript; split: valid
- Lengths: docstring 111 chars / 21 tokens; code 1544 chars / 409 tokens
- Machine indicators: function names ['set']; names in docstring ['set']; token overlap 0.115; comment ratio 0.018; repeated-line ratio 0.154; bundle []; minified False
- Assessment: Strong alignment — docstring describes caption track selection; code sets captions.
- Docstring excerpt:
~~~text
Set captions by track index Used internally for the currentTrack setter with the passive option forced to false
~~~
- Code excerpt:
~~~text
function set(index) {
	    var passive = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : true;
	    var tracks = captions.getTracks.call(this); // Disable captions if setting to -1

	    if (index === -1) {
	      captions.toggle.call(this, false, passive);
	      return;
	    }

	    if (!is$1.numb
~~~

## 4. Pair-alignment review

Machine-detected indicators (counts over all 413,597 allowed records):

| Indicator | Count | Interpretation |
|---|---:|---|
| Empty docstring | 0 | No missing docstrings in the allowed corpus. |
| Empty code | 0 | No missing code in the allowed corpus. |
| Duplicate docstrings | 21,676 extra occurrences | Repeated boilerplate/stub descriptions; may weaken alignment. |
| Non-ASCII docstrings | 10,565 | Multilingual or formula-heavy documentation; not automatically invalid. |
| Bundle/minified markers | 4,349 | Likely bundled or minified source; poor function-level training signal. |
| Wrapper/decorator-heavy | 22,785 | May be framework boilerplate; quality depends on the example. |
| Comment-heavy code | 1,221 | Documentation-heavy snippets; may contain little executable logic. |
| High repetition | 31,674 | Repeated code fragments; may overrepresent patterns. |

Qualitative findings from the 90 sampled records:

- Most Python samples are coherent function-level examples whose docstrings describe the implemented behavior, including parameters and return values.
- A minority of Python samples have very short or vague docstrings (for example, `":return: int"`, `"Handle received data."`) that provide little semantic guidance.
- JavaScript samples are more mixed: many are coherent, but a larger share have placeholder, banner, license-header, or extremely short docstrings.
- Some sampled JavaScript records are bundled/minified or contain license headers; these are not clean function-level training examples.
- No sampled record showed a docstring that directly contradicts the code, but several provided only a name or a section header rather than a description.
- Code is generally complete in the sampled records; apparent truncation in the audit is mostly due to excerpt limits, not dataset truncation. The corpus builder enforces minimum code length.

## 5. Training suitability (sampled records)

| Category | Count | Share of 90 sampled |
|---|---:|---:|
| Strong description/code alignment | 61 | 67.8% |
| Partial alignment | 18 | 20.0% |
| Weak or unclear alignment | 7 | 7.8% |
| Apparently mismatched or unusable | 4 | 4.4% |
| Total | 90 | 100.0% |

Supporting examples:

- Strong: `221ec4bc67a8027b` (Python `from_function`), `77a9684dd3a06ce4` (Python `delete_role`), `e60a07913bf36c80` (JavaScript `closest`), `33d94a8d3d599972` (JavaScript stdin processing).
- Partial: `7eab2b3ce98b2f1c` (Python accessor with only a return note), `b421b8324fd24a06` (JavaScript transform-stream controller), `ca5186d717a13cae` (status wrapper).
- Weak: `60fd8d6f72c66754` (vague "Handle received data."), `2ead5044608a05a2` ("create custom message for validation"), `575d018afba2f151` ("private").
- Mismatched/unusable: `25769d4927cf34f4` (Angular license header), `b4df2d5bec202438` (docstring is an import source), `d1148fd1d7a536c7` (section banner).

These proportions are sample estimates, not dataset-wide labels. They suggest that description/code alignment is generally good for Python function-level examples, more variable for JavaScript, and degraded by boilerplate, banners, and generated stubs. No filtering, deletion, or rewriting was performed.

## 6. Training configuration review

Source: `configs/training.yaml`, `configs/model.yaml`, Phase 5B/5C reports, `logs/training/first_pass_metrics.csv`, and `logs/training/first_pass_validation.json`.

| Item | Value |
|---|---|
| Training records (code records) | 392,632 (290,463 Python + 102,169 JavaScript) |
| Training tokens (serialized stream) | 133,192,957 |
| Validation records | 20,965 (14,671 Python + 6,294 JavaScript) |
| Validation tokens | 7,756,800 |
| Blocks (sequences of 512 tokens) | 260,142 training; 15,150 validation |
| Micro-batch size | 4 sequences |
| Gradient accumulation | 8 |
| Effective batch | 32 sequences = 16,384 tokens |
| Context length | 512 |
| Optimizer | AdamW, lr 3e-4, weight decay 0.1, grad clip 1.0 |
| Schedule | 8,130 optimizer steps, warmup 163, cosine decay to 10% of base LR |
| Passes over training stream | 1 complete pass (all 260,142 blocks consumed exactly once) |
| Initial training loss | 9.774015 (step 1) |
| Final training loss | 2.143797 (step 8,130) |
| Initial validation loss | 9.791127 (step 0, 64-block subset) |
| Final validation loss (subset) | 2.392860 (step 8,130, 64 blocks) |
| Final validation loss (full split) | 2.446182 (step 8,130, 15,150 blocks) |
| Validation perplexity (full split) | 11.544 |
| Intended schedule completed | Yes — 8,130/8,130 steps; final partial group of 14 blocks was normalised by actual sequence count |

The training stream mixes code records and Python documentation. Loss is not masked at `<|eos|>` separators, so a small fraction of targets are structural boundaries. The model completed the intended single pass; no claim is made that another pass would improve generation.

## 7. Conclusions

- The allowed training/validation corpus is large, structurally consistent, and has no empty docstrings or code fields.
- Alignment is generally strong for Python function-level examples, weaker and more variable for JavaScript, and degraded by boilerplate, banners, license headers, and generated stubs.
- Repetition, long files, and bundled/minified source are present in small but non-negligible proportions and may contribute to poor generation by overrepresenting patterns or introducing non-function-level text.
- Multilingual and formula-heavy docstrings are common and should not be treated as invalid without semantic review.
- Training completed its intended single pass with monotonically decreasing loss; the available evidence does not establish that data quality alone explains Phase 6 generation results, nor that additional training would improve them.

## 8. Files inspected

- `data/processed/polycodegpt_corpus/train.jsonl`
- `data/processed/polycodegpt_corpus/valid.jsonl`
- `configs/model.yaml`
- `configs/training.yaml`
- `reports/phase5b_training_pilot_report.md`
- `reports/phase5c_first_training_pass_report.md`
- `logs/training/first_pass_metrics.csv`
- `logs/training/first_pass_validation.json`
- `logs/training/first_pass_preflight.json`
- `src/polycodegpt/tokenization/serializer.py`
- `src/polycodegpt/data/tokenizer_io.py`

## 9. Tests run

- No new tests were added; this phase is a read-only audit.
- Existing unit suite was not modified.
- Data boundaries were enforced by reading only the two allowed JSONL files.

## 10. Exact data boundaries respected

- Read: `train.jsonl`, `valid.jsonl`.
- Not read: `test.jsonl`, `test.bin`, or any test metrics.
- No dataset, checkpoint, tokenizer, configuration, or report other than this audit report was modified.
