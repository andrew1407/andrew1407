# Python Style

Repos: [ReignForce/TrainingServer](https://github.com/Andrew1407/ReignForce/tree/main/TrainingServer) &
[CartAndPole/TrainigServer](https://github.com/Andrew1407/CartAndPole) (TF, supervised + RL, WS/UDP servers),
[expression_parser](https://github.com/Andrew1407/expression_parser) (lexer/AST/transform pipeline, the canonical
architecture reference), [ImageUpscaler](https://github.com/Andrew1407/ImageUpscaler) (TF + Telegram bot),
[EmbeddedSystems](https://github.com/Andrew1407/EmbeddedSystems) (signal/DFT/scheduling labs),
[DataScience](https://github.com/Andrew1407/DataScience), [AI_basics](https://github.com/Andrew1407/AI_basics).

Read [common.md](./common.md) first.

---

## Module taxonomy

- Every multi-module project replicates: a `containers.py`/`datastructs.py` per domain (only `@dataclass`
  + constants), a `utils.py` per package (pure functions only), a top `main.py` that wires deps and holds
  `if __name__ == '__main__':`, an isolated `plots.py`/output module when visualization is needed.
- `expression_parser` is the reference layout: `parser/` (lexer) → `analyzer/` (AST) →
  `equivalent_forms/`, `parallel_tree/`, `conveyor_simulation/` (transforms) → `tree_output/` (rendering),
  with a top-level `expression_data_builder.py` orchestrator and a thin `main.py`.
- `__init__.py` almost always empty; non-empty only to re-export package entry points.

## Naming

- Files lowercase_underscored, role-suffixed (`_parser`, `_analyzer`, `_builder`, `_trainer`,
  `_predictor`, `_output`, `_converter`, `_generator`). Labs are flat `N-lab.py`.
- Classes PascalCase with role suffix; exceptions end in `Exception`.
- Functions/methods snake_case. **Private methods use double-underscore `__name`** (not single) — the default.
- Public read accessors `get_*()` returning **immutable `tuple` copies**.
- Static "named constructor" methods `of(...)` (`Token.of`, `DynamicConveyor.of`).
- Top-level pure functions are imperative verbs (`build_tree_graph`, `minimize_depth`, `analyze`).
- Module constants `SCREAMING_SNAKE_CASE`; paths via f-string off a base dir.

## OOP vs functional (dual track)

- **Stateful classes** for encapsulated mutable state: do all work in `__init__`, expose results only via
  `get_*()` (returning tuples). (`ExpressionParser`, `SyntaxAnalyzer`, `DynamicConveyor`, RL trainers, servers.)
- **Pure-function modules** for all tree/data transformations — no classes, just public functions +
  `__module_private` helpers.
- **Orchestrator classes** wire functional modules together (`ExpressionDataBuilder`); `ExpressionView` is a
  facade of stub methods decorated to inject IO side effects.

## Type hints

- Comprehensive on public interfaces. **Distinctive: `(Type | NoneType)` unions** via
  `from types import NoneType` (not `Optional` / `| None`).
- Lowercase builtins (`list[Token]`, `tuple[Token, ...]`, `dict[str,int]`), never `typing.List/Dict/Tuple`;
  `typing` only for `Iterable`/`Callable`/`Sequence`.
- **Named type aliases at module scope** for any compound type used 2+ times
  (`NodesTuple = tuple[Node, ...]`, `FramePair = tuple[pd.DataFrame, pd.DataFrame]`).

## Dataclasses / enums / no ABCs

- `@dataclass` heavily preferred for containers, freely mixed with methods (`to_json()`) and static `of(...)`.
- AST nodes are a **`@dataclass` inheritance hierarchy** (`Node` → `UnaryOperatorNode`/`BinaryOperatorNode`/
  `FunctionNode`) — **no ABCs anywhere**, no `abstractmethod`.
- `class X(Enum)`; `class X(str, Enum)` for direct string comparison; plain class-as-namespace for
  non-enumerable constant groups. `namedtuple(typename=..., field_names=...)` (explicit kwargs) for RL `Experience`.

## `match`/`case` — the signature idiom

Used pervasively for all type dispatch (AST traversal, token/character dispatch) instead of `isinstance`
chains or a visitor class. Deep destructuring patterns with `as` binding:
```python
match node:
  case BinaryOperatorNode(value=Token(value=Operator.PLUS.value)): ...
  case UnaryOperatorNode(): ...
  case Node(): ...
  case _: ...
```

## Parser / AST architecture (reference)

- Lexer: stateful class, **collects** errors into a list then raises `ExceptionGroup`; char dispatch via
  `match`; returns immutable tuples.
- AST: plain `@dataclass` hierarchy, dispatch by `match` not visitor; `to_json()` on base for debug.
- Recursive-descent parser: `__build_additive → __build_multiplicative → __build_unary → __build_primary
  → __build_function`; single `__get_current_token(next=False)` accessor; raises domain exceptions immediately.
- Transforms: pure recursive module functions; `deepcopy` for non-mutating tree passes; one named function per pass.

## ML / RL pipelines

- Models = free factory functions returning `keras.Sequential` (typical stack `Dense(128,'relu')→64→32→out`);
  no `keras.Model` subclassing for simple nets.
- `build_model(data)` does split/compile/fit/evaluate/save, returns `(model, mse)`; `load_model()` a one-liner
  beside it. Hyperparameters are module-level constants, not config files.
- RL: `Experience` namedtuple, `ReplayMemory` over `deque`, callable `EpsilonGreedyStrategy` (`__call__` +
  `@staticmethod` calc), explicit `tf.GradientTape`, separate policy/target nets via `set_weights`.
- Servers: `asyncio.DatagramProtocol` subclass, state in `__init__`, message dispatch via `startswith`
  chains, separate `parse_inputs(data)` function, `make_*()` factory + async `main()` wiring.
- Custom Keras layers subclass `Layer` (`build`/`call`/`get_config`), split logic into `@tf.function` methods.

## Decorators

- Guard/precondition decorators (`dist_check` ensures output dir exists).
- Parameterized decorators returning `decorator(fn)→wrapper` (`parameters_wrapper(sort_key)`).
- Unique method-wrapper decorator injecting pre/post side effects (`ExpressionView.__method_wrapper`).
- A separate `loggers.py` with **one logger decorator per public function** in the paired logic module.

## Formatting & idioms

- **2-space indentation** (distinctive, non-PEP8) — use it. (`CartAndPole/TrainigServer` is 4-space; outlier.)
- Inline single-expression statements on the control-flow line (`if next: self.__position += 1`,
  `for t in tokens: t.value = t.value.strip()`).
- Single blank line between methods.
- `list()` not `[]` for empty lists; return `tuple(...)` for immutability.
- Module-scope lambdas for simple transforms (`edf_sort = lambda t: (t.deadline, t.arrival, t.wcet)`).
- Dict-as-dispatch (`ALGORITHMS = dict(fifo=..., edf=...)`); `partial(...)` for currying; f-strings;
  `del df['col']` over `drop`.

## Errors, tests, entry points

- Domain exceptions at top of their module, `super().__init__(message.format(...))`, store context attrs.
  Lexer collects → `ExceptionGroup`; parser raises immediately; REPL/servers catch `(KeyboardInterrupt, EOFError)`.
- Tests: **`unittest.TestCase`** (never pytest), one class per domain, `test_<scenario>` names, build expected
  tuples with `Token.of(...)`, `assertTupleEqual`; CI runs `python -m unittest`.
- Entry point: module constants → `make_*()` factory (builds all components) → minimal
  `if __name__ == '__main__':` calling one lifecycle method.
