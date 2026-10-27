//! Python module `zero._engine` (built by `tools/build_py.sh` into `python/zero/_engine.pyd`):
//! the core's toy games only (`toy`, and `toy_team`: deferred results, teams). A game's module
//! looks the same with its own games added.
zero_py::engine_module!(_engine, {
    "toy" => zero_search::toy::ToyGame,
    "toy_team" => zero_search::toy_team::ToyTeam,
});
