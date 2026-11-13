//! Python module `zero._engine` (built by `tools/build_py.sh` into `python/zero/_engine.pyd`):
//! the core's toy games only (`toy`; `toy_team`: deferred results, teams; `toy_tail`: variable-length rows,
//! docs/search.md 5.3; `toy_match`: chained episodes). A game's module
//! looks the same with its own games added.
zero_py::engine_module!(_engine, {
    "toy" => zero_search::toy::ToyGame,
    "toy_team" => zero_search::toy_team::ToyTeam,
    "toy_tail" => zero_search::toy_tail::TailGame,
    "toy_match" => zero_search::toy_match::ToyMatch,
});
