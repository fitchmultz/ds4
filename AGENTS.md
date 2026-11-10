# Fork rules

This repository is Mitch's fork of [antirez/ds4](https://github.com/antirez/ds4).
`AGENT.md` is the upstream guide and applies to all code here. This file holds
the rules that exist only in the fork.

## Keep `main` synced with upstream

`main` is upstream `main` plus the fork's commits. Sync it before you start
work in the fork and before you build a runtime from it:

```sh
git fetch https://github.com/antirez/ds4.git main
git merge FETCH_HEAD
```

- Merge upstream into `main`. Do not rebase or force-push `main`.
- After every sync, run the checks in the next section before you push.

## Keep image conversations in the caches

The fork adds one capability to upstream: a conversation that contains an image
keeps its cache.

- A live session with an image still matches by text when its tokens differ
  from the request (`request_cache_text()` and `checkpoint_cache_text()` in
  `ds4_server.c`).
- The disk KV cache saves and restores a conversation with images
  (`DS4_KVSTORE_EXT_VISION` in `ds4_kvstore.c`,
  `ds4_session_load_payload_multimodal()` in `ds4.c`).

Every sync must leave this capability working. On a Mac, where the build is
the Metal one, run:

```sh
make ds4-server ds4_test
./ds4_test --server
make test-session-state
```

With GLM 5.3 Flash weights and the GLM vision encoder, and no other large
model loaded, also run the test that restarts the server. It needs a new
output directory for each run:

```sh
python3 tests/test_server_vision_disk.py MODEL.gguf \
	--vision VISION_ENCODER.gguf --output "$(mktemp -d)"
```

When a merge conflicts with the fork's code, resolve it so that these checks
pass. When upstream gains the same capability, take the upstream code, delete
the fork's version of that part, and keep `tests/test_server_vision_disk.py`
passing.

Write fork commits the way `AGENT.md` and `CONTRIBUTING.md` ask, so that each
one can go upstream unchanged.
