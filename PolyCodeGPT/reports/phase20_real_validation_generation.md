# PolyCodeGPT Phase 20 — Real Validation Generation

- Checkpoint step: 30500
- Device: cuda
- Precision: BF16
- Decoding: Greedy
- Seed: 20261119
- Test partition opened: No

> Exact match and token similarity measure reference reproduction, not semantic correctness.

## Summary

| Language | Samples | Exact matches | Mean token similarity | EOS rate |
|---|---:|---:|---:|---:|
| python | 8 | 0 | 0.475 | 0.250 |
| javascript | 8 | 0 | 0.238 | 0.125 |

## Individual results

### python — sample 1

- Record: `2773a538562966619d4dce19402e062f`
- Source: `lmnotify/lmnotify.py`
- Exact match: False
- Token similarity: 0.378
- Generated/reference tokens: 90/90
- EOS generated: False
- Python syntax valid: True

**Generated:**

```python
def save_devices(self):
        """save devices that have been obtained from LaMetric cloud
        to a local file
        """
        if self.device_id == self.device_id:
            self.device_id = self.device_id
        else:
            self.device_id = self.device_id
        self.device_id = self.device_id
        self.device_id = self.device_
```

**Reference:**

```python
def save_devices(self):
        """
        save devices that have been obtained from LaMetric cloud
        to a local file
        """
        log.debug("saving devices to ''...".format(self._devices_filename))
        if self._devices != []:
            with codecs.open(self._devices_filename, "wb", "utf-8") as f:
                json.dump(self._devices, f)
```

### python — sample 2

- Record: `9ad77be25aafdce6c56ab55e93c3bbe7`
- Source: `pygeotools/lib/geolib.py`
- Exact match: False
- Token similarity: 0.327
- Generated/reference tokens: 113/113
- EOS generated: False
- Python syntax valid: True

**Generated:**

```python
def localortho(lon, lat):
    """Create srs for local orthographic projection centered at lat, lon"""
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split(' ')
    lat = lat.split
```

**Reference:**

```python
def localortho(lon, lat):
    """Create srs for local orthographic projection centered at lat, lon
    """
    local_srs = osr.SpatialReference()
    local_proj = '+proj=ortho +lat_0=%0.7f +lon_0=%0.7f +datum=WGS84 +units=m +no_defs ' % (lat, lon)
    local_srs.ImportFromProj4(local_proj)
    return local_srs
```

### python — sample 3

- Record: `704cfb72f41c8fd98167e3343fe370f8`
- Source: `redisco/containers.py`
- Exact match: False
- Token similarity: 0.661
- Generated/reference tokens: 59/59
- EOS generated: False
- Python syntax valid: False

**Generated:**

```python
def sunion(self, *other_sets):
        """
        Union between Sets.

        Returns a set of common members. Uses Redis.sunion.
        """
        if self.sunion is not None:
            return self.sunion(self.sunion)
        else:
```

**Reference:**

```python
def sunion(self, *other_sets):
        """Union between Sets.

        Returns a set of common members. Uses Redis.sunion.
        """
        return self.db.sunion([self.key] + [s.key for s in other_sets])
```

### python — sample 4

- Record: `07ff8bc39df0ed96cd68d298ebd77f8a`
- Source: `src/redis_lock/__init__.py`
- Exact match: False
- Token similarity: 0.429
- Generated/reference tokens: 31/53
- EOS generated: True
- Python syntax valid: True

**Generated:**

```python
def reset(self):
        """Forcibly deletes the lock."""
        self.lock.reset()
        self.lock.reset()
```

**Reference:**

```python
def reset(self):
        """
        Forcibly deletes the lock. Use this with care.
        """
        _eval_script(self._client, RESET, self._name, self._signal)
        self._delete_signal()
```

### python — sample 5

- Record: `ebbbcb370eca10ba1003dfac9cac8f83`
- Source: `pyramid_zipkin/request_helper.py`
- Exact match: False
- Token similarity: 0.458
- Generated/reference tokens: 168/168
- EOS generated: False
- Python syntax valid: False

**Generated:**

```python
def should_not_sample_path(request):
    """Decided whether current request path should be sampled or not. This is
    checked previous to `should_not_sample_route` and takes precedence.

    :param: current active pyramid request
    :returns: boolean whether current request path is blacklisted.
    """
    if request.path.exists(request.path):
        return False
    if request.path.exists(request.path):
        return False
    if request.path.exists(request.path):
        return False
    if request.path.exists(request.path):
        return False
    if request.path.exists(request.path):
        return False
    if request.path.exists(request.path):
        return False
    if request.path.exists(request
```

**Reference:**

```python
def should_not_sample_path(request):
    """Decided whether current request path should be sampled or not. This is
    checked previous to `should_not_sample_route` and takes precedence.

    :param: current active pyramid request
    :returns: boolean whether current request path is blacklisted.
    """
    blacklisted_paths = request.registry.settings.get(
        'zipkin.blacklisted_paths', [])
    # Only compile strings, since even recompiling existing
    # compiled regexes takes time.
    regexes = [
        re.compile(r) if isinstance(r, six.string_types) else r
        for r in blacklisted_paths
    ]
    return any(r.match(request.path) for r in regexes)
```

### python — sample 6

- Record: `d337f5e645e780374110e30ea05ea5a2`
- Source: `fluent_contents/extensions/pluginpool.py`
- Exact match: False
- Token similarity: 0.725
- Generated/reference tokens: 42/60
- EOS generated: True
- Python syntax valid: True

**Generated:**

```python
def get_model_classes(self):
        """Return all :class:`~fluent_contents.models.ContentItem` model classes which are exposed by plugins."""
        return self._model_classes
```

**Reference:**

```python
def get_model_classes(self):
        """
        Return all :class:`~fluent_contents.models.ContentItem` model classes which are exposed by plugins.
        """
        self._import_plugins()
        return [plugin.model for plugin in self.plugins.values()]
```

### python — sample 7

- Record: `94f730daa536ae3771a8ab10f2bd5a1b`
- Source: `bakery/management/commands/build.py`
- Exact match: False
- Token similarity: 0.183
- Generated/reference tokens: 164/164
- EOS generated: False
- Python syntax valid: False

**Generated:**

```python
def build_media(self):
        """Build the media files."""
        media_files = self.get_media_files()
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.append(media_files)
        media_files.
```

**Reference:**

```python
def build_media(self):
        """
        Build the media files.
        """
        logger.debug("Building media directory")
        if self.verbosity > 1:
            self.stdout.write("Building media directory")
        if os.path.exists(self.media_root) and settings.MEDIA_URL:
            target_dir = path.join(self.fs_name, self.build_dir, settings.MEDIA_URL.lstrip('/'))
            logger.debug("Copying {}{} to {}{}".format("osfs://", self.media_root, self.fs_name, target_dir))
            copy.copy_dir("osfs:///", smart_text(self.media_root), self.fs, smart_text(target_dir))
```

### python — sample 8

- Record: `099edd4601509535dd780f95bec218a9`
- Source: `setup.py`
- Exact match: False
- Token similarity: 0.642
- Generated/reference tokens: 106/106
- EOS generated: False
- Python syntax valid: False

**Generated:**

```python
def extract_version(filepath='jeni.py', name='__version__'):
    """Parse __version__ out of given Python file.

    Given jeni.py has dependencies, `from jeni import __version__` will fail.
    """
    if name in os.path.exists(filepath):
        raise ValueError('The name must be a string')
    if name in os.path.exists(filepath):
        raise ValueError('The name must be a string')
    if name in os
```

**Reference:**

```python
def extract_version(filepath='jeni.py', name='__version__'):
    """Parse __version__ out of given Python file.

    Given jeni.py has dependencies, `from jeni import __version__` will fail.
    """
    context = {}
    for line in open(filepath):
        if name in line:
            exec(line, context)
            break
    else:
        raise RuntimeError('{} not found in {}'.format(name, filepath))
    return context[name]
```

### javascript — sample 1

- Record: `fbc34ade7f69ef7a10da22d0241244da`
- Source: `src/queries/label-text.js`
- Exact match: False
- Token similarity: 0.152
- Generated/reference tokens: 164/164
- EOS generated: False

**Generated:**

```javascript
function getAllByLabelText(container, text, ...rest) {
  const { label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label
```

**Reference:**

```javascript
function getAllByLabelText(container, text, ...rest) {
  const els = queryAllByLabelText(container, text, ...rest)
  if (!els.length) {
    const labels = queryAllLabelsByText(container, text, ...rest)
    if (labels.length) {
      throw getElementError(
        `Found a label with the text of: ${text}, however no form control was found associated to that label. Make sure you're using the "for" attribute or "aria-labelledby" attribute correctly.`,
        container,
      )
    } else {
      throw getElementError(
        `Unable to find a label with the text of: ${text}`,
        container,
      )
    }
  }
  return els
}
```

### javascript — sample 2

- Record: `c67f254b6e3ffb998f8af2341660ed37`
- Source: `src/client/gmeServerRequests.js`
- Exact match: False
- Token similarity: 0.166
- Generated/reference tokens: 199/199
- EOS generated: False

**Generated:**

```javascript
function checkMetaRules(nodePaths, includeChildren, callback) {
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0) {
    return;
  }
  if (nodePaths.length === 0
```

**Reference:**

```javascript
function checkMetaRules(nodePaths, includeChildren, callback) {
            var parameters = {
                command: CONSTANTS.SERVER_WORKER_REQUESTS.CHECK_CONSTRAINTS,
                checkType: 'META', //TODO this should come from a constant
                includeChildren: includeChildren,
                nodePaths: nodePaths,
                commitHash: state.commitHash,
                projectId: state.project.projectId
            };

            storage.simpleRequest(parameters, function (err, result) {
                if (err) {
                    logger.error(err);
                }

                if (result) {
                    client.dispatchEvent(client.CONSTANTS.META_RULES_RESULT, result);
                } else {
                    client.notifyUser({
                        severity: 'error',
                        message: 'Evaluating Meta rules failed with error.'
                    });
                }

                if (callback) {
                    callback(err, result);
                }
            });
        }
```

### javascript — sample 3

- Record: `cc62a70b803e50f71e77671d12ba2abf`
- Source: `lib/shared.js`
- Exact match: False
- Token similarity: 0.222
- Generated/reference tokens: 126/126
- EOS generated: False

**Generated:**

```javascript
function match(self, dirname, basename, callback) {
        var path = path.join(dirname, basename);
        if (path.indexOf(path) > -1) {
            path = path.join(dirname, basename);
        }
        if (path.indexOf(path) > -1) {
            path = path.join(dirname, basename);
        }
        if (path.indexOf(path) > -1) {
            path = path.join(dirname, basename);
        }
        if (path.indexOf(path) > -1) {
            path = path.join(
```

**Reference:**

```javascript
function match(self, dirname, basename, callback) {
  var pathname, stats, pattern, matches;

  pattern = patternFor(self, basename);
  matches = self.entries(dirname).filter(function(m) { return pattern.test(m); });
  matches = sortMatches(self, matches, basename);

  while (matches.length) {
    pathname  = path.join(dirname, matches.shift());
    stats     = self.stat(pathname);

    if (stats && stats.isFile() && callback(pathname)) {
      return pathname;
    }
  }
}
```

### javascript — sample 4

- Record: `71ab9c52c49fddace9d33d9cbe5ed10d`
- Source: `src/js/plugins/youtube.js`
- Exact match: False
- Token similarity: 0.131
- Generated/reference tokens: 84/84
- EOS generated: False

**Generated:**

```javascript
function parseId(url) {
  var url = url.replace(/\\/[^\\/]+$/, '');
  var url = url.replace(/\\/[^\\/]+$/, '');
  var url = url.replace(/\\/[^\\/]+$/, '');
  var url = url.replace(/\\/[^\\/]+$/, '');
  var url = url.replace(/\\/[^\\/]+$/, '');
  var url = url.replace
```

**Reference:**

```javascript
function parseId(url) {
    if (is.empty(url)) {
        return null;
    }

    const regex = /^.*(youtu.be\\/|v\\/|u\\/\\w\\/|embed\\/|watch\\?v=|&v=)([^#&?]*).*/;
    return url.match(regex) ? RegExp.$2 : url;
}
```

### javascript — sample 5

- Record: `e2d7c5c2bc73b291d0438608c40a1f1b`
- Source: `src/resolver.js`
- Exact match: False
- Token similarity: 0.323
- Generated/reference tokens: 65/65
- EOS generated: False

**Generated:**

```javascript
function _isList(gqlTye) {
  return gqlTye.find(gqlTye.find(gqlTye.find(gqlTye.find(gqlTye.find(gqlTye.find(gqlTye.find(gqlT
```

**Reference:**

```javascript
function _isList(gqlTye) {
  if (gqlTye instanceof GraphQLList) {
    return true;
  }

  if (gqlTye instanceof GraphQLNonNull) {
    return _isList(gqlTye.ofType);
  }

  return false;
}
```

### javascript — sample 6

- Record: `7e82a51b4a72a68932535886dc014e75`
- Source: `lib/shared-mongo-connections.js`
- Exact match: False
- Token similarity: 0.439
- Generated/reference tokens: 66/66
- EOS generated: False

**Generated:**

```javascript
function getMongooseConnection(mongoDbUrl, logger, cb) {
  var connection = new mongoose.Connection(mongoDbUrl, logger, cb);
  connection.on('connection', function(err) {
    if (err) {
      console.log(err);
      console.log(err);
      console
```

**Reference:**

```javascript
function getMongooseConnection(mongoDbUrl, logger, cb) {
  logger.debug("creating mongoose connection for data_source_update job", {mongoDbUrl: mongoDbUrl});
  var mongooseConnection = mongoose.createConnection(mongoDbUrl);
  return cb(undefined, mongooseConnection);
}
```

### javascript — sample 7

- Record: `71ee8117ce2a0834def787b277a2c0a6`
- Source: `src/util/WWMath.js`
- Exact match: False
- Token similarity: 0.333
- Generated/reference tokens: 39/39
- EOS generated: True

**Generated:**

```javascript
function(value) {
        var value = value.split('.');
        if (value.length > 0) {
            value = value.split('.');
        }
        return value;
    }
```

**Reference:**

```javascript
function (value) {
                var power = Math.floor(Math.log(value) / Math.log(2));
                return Math.pow(2, power);
            }
```

### javascript — sample 8

- Record: `7249ecbfd564d411fa9819b4a241d5f9`
- Source: `lib/functions/authenticate.js`
- Exact match: False
- Token similarity: 0.137
- Generated/reference tokens: 124/124
- EOS generated: False

**Generated:**

```javascript
function getTokenCustom(code, oauth2Client) {
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
  var token = token.token;
```

**Reference:**

```javascript
function getTokenCustom(code, oauth2Client) {
    return new Promise((resolve, reject) => {
        if (code) {
            oauth2Client.getToken(code, (err, newToken) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(newToken);
                return;
            });
        } else {
            oauth2Client.refreshAccessToken((err, newToken) => {
                if (err) {
                    reject(err);
                }
                resolve(newToken);
                return;
            });
        }
    });
}
```
