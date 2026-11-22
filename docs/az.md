

**Measured** (the desktop beside two live runs and other sessions' gates, 2026-11-22). The whole suite in four shards
with the GPU visible: 904 s (shards of 219 to 220 tests; the longest 793 s of pytest) against 44 and 52 minutes in one
process the same day - 829 passed, 31 skipped, two `test_remoteplay` tests failed beside the others and passed alone,
and one failure that is not the shards': `tests/test_dup_reduce.py::test_max_is_deterministic_on_cuda_and_matches_the_cpu`
fails alone too on this commit with the GPU visible (MTG Zero's gate hides the GPU, so it had not been run there). With
the GPU hidden, MTG Zero's gate ran the same suite in four shards of its own: 780 passed, 6.2 to 12.7 minutes of test
time a shard. `SKIP_CORE=1 bash tools/smoke_all.sh --with-this-core --build-only`, three games at once: GREEN, 3 of 3
checked, the builds 40 / 42 / 89 s (171 s one after the other) in 96 s for the whole command.
