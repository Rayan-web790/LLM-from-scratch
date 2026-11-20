# Phase 11 — Sandbox Containment Verification

Docker server: `29.0.1` (linux containers). Probes: 16/16 passed.

## Container configuration

```
--network none
--read-only
--tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m
--user 10001:10001
--cap-drop ALL
--security-opt no-new-privileges:true
--memory 128m --memory-swap 128m
--cpus 0.50
--pids-limit 32
--ulimit nofile=64
--ulimit fsize=1048576
no bind mounts; payload over stdin, results over stderr
container removed after every run
```

## Probe results

| probe | property | expected | observed | verdict |
|---|---|---|---|---|
| `py_net_tcp` | no network (TCP connect) | blocked | BLOCKED OSError [Errno 101] Network is unreachable | PASS |
| `py_net_dns` | no network (DNS) | blocked | BLOCKED gaierror [Errno -3] Temporary failure in name resolution | PASS |
| `py_fs_root_write` | read-only root filesystem | blocked | BLOCKED /opt/harness/probe.txt OSError 30 \| BLOCKED /etc/probe.txt OSError 30 \| BLOCKED /harness_probe.txt OSError 30 | PASS |
| `py_host_paths` | no host project files visible | blocked | FOUND [] \| ROOT_ENTRIES ['.dockerenv', 'bin', 'boot', 'dev', 'etc', 'home', 'lib', 'lib64', 'media', 'mnt', 'opt', 'proc', 'root', 'run', 'sbin', 'srv', 'sys', 'tmp', 'usr', 'var' | PASS |
| `py_mounts` | no host bind mounts | blocked | HOST_MOUNTS 0 \| ROOT_LINE 449 335 0:83 / / ro,relatime - overlay overlay rw,lowerdir=/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/870/fs:/va | PASS |
| `py_identity` | non-root user | allowed | UID 10001 \| EUID 10001 \| CAPEFF 0000000000000000 \| NONEWPRIVS 1 \| SECCOMP 2 | PASS |
| `py_timeout` | execution timeout | killed | killed by host timeout | PASS |
| `py_memory` | memory ceiling | blocked | oom_killed=True exit=137 | PASS |
| `py_pids` | process ceiling | blocked | BLOCKED fork after 20 BlockingIOError \| BLOCKED fork after 5 BlockingIOError \| BLOCKED fork after 4 BlockingIOError \| BLOCKED fork after 6 BlockingIOError \| BLOCKED fork after  | PASS |
| `py_fsize` | file-size ceiling | blocked | BLOCKED OSError 27 | PASS |
| `js_net_fetch` | no network (fetch) | blocked | BLOCKED TypeError ENETUNREACH | PASS |
| `js_net_tcp` | no network (TCP) | blocked | BLOCKED ENETUNREACH \| BLOCKED timeout | PASS |
| `js_fs_root_write` | read-only root filesystem | blocked | BLOCKED /opt/harness/probe.txt EROFS \| BLOCKED /etc/probe.txt EROFS | PASS |
| `js_identity` | non-root user | allowed | UID 10001 GID 10001 | PASS |
| `js_timeout` | execution timeout | killed | killed by host timeout | PASS |
| `js_memory` | memory ceiling | blocked | oom_killed=True exit=137 | PASS |

## Fixed limits

- memory: 128m (swap disabled at the same value)
- cpus: 0.50
- pids: 32
- open files: 64
- max file size: 1048576 bytes
- wall clock: 10s per execution, host-enforced by kill

## Residual risk

- The container shares the Docker Desktop Linux VM kernel; a kernel exploit is theoretically exploitable. No seccomp profile beyond Docker's default and no user namespace remapping is configured.
- Docker Desktop itself runs with the host user's privileges; container escape flaws in Docker are out of scope of the app-level sandbox.
- `--tmpfs /tmp` is writable (16 MiB, noexec) by design; probes confirm root and `/opt` are not writable.
- Probes ran before any model-generated code was executed.
