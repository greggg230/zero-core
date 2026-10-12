//! Python module `zero._engine` (built by `tools/build_py.sh` into `python/zero/_engine.pyd`):
//! the core's toy game only. A game's module looks the same with its own games added.
zero_py::engine_module!(_engine, {
    "toy" => zero_search::toy::ToyGame,
});
