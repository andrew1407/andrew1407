# Common Coding Style

Platform-agnostic architecture, naming, and decomposition patterns that recur across this
author's repos in **every** language (JS/TS, C++, Unreal C++, Unity C#, Go, Python, Kotlin).
Use this as the baseline when generating new code in any language; then layer the
language-specific file on top.

> Scope: only the patterns that are *distinctive and repeated* across languages are listed
> here. Obvious/standard syntax and well-known framework boilerplate are intentionally omitted.

---

## 1. The signature architecture: Interface → Strategy → Factory

This triad is the single most recognizable cross-language fingerprint. It shows up in JS
([gallows-remastered](https://github.com/Andrew1407/gallows-remastered), [Domino](https://github.com/Andrew1407/Domino),
[DichBox](https://github.com/Andrew1407/DichBox)), Unreal C++ ([ClientConnectionStrategies](https://github.com/Andrew1407/ClientConnectionStrategies)),
Unity C# ([GallowsGame](https://github.com/Andrew1407/GallowsGame)), Kotlin ([DichBoxMobile](https://github.com/Andrew1407/DichBoxMobile)),
and Python ([expression_parser](https://github.com/Andrew1407/expression_parser), RL servers).

- **A narrow interface** defines a capability contract (1–4 methods). Examples that recur with
  almost identical shape across languages: `IClient` / `IConnection` (UE C++), `IGameplayStrategy`
  (Unity), `StorageClient` (TS), `INneModel` (UE), `IClient`/`IConnection`.
- **Multiple concrete strategies** implement that one interface — typically one per protocol,
  per algorithm, or per backend.
- **A factory** assembles/selects the strategy by a label or key, usually a string/enum:
  - JS: `getFatofies(labels)`, dynamic `import('servers/<label>.js')`, `BoxesControllerFactory`.
  - Unity C#: `GameplayStrategyFactory` holding `Dictionary<string, Func<Uri, IGameplayStrategy>>`.
  - UE C++: `UClientFactory` with `CreateHttpClient`/`CreateWsClient` + templated `CreateClient<T>`.
  - Go: `Create*`/`New*` returning an interface, concrete type unexported.
  - Python: `make_*()` factory functions.
  - Kotlin: `*API` concrete over `ApiParser<C>` abstract base.

**When generating new code:** if there are 2+ interchangeable behaviors, define an interface,
put each behavior in its own file/class, and add a factory keyed by an enum/string label —
do **not** inline a `switch` over behaviors.

---

## 2. Recurring domain: multi-protocol client/server (HTTP / WS / UDP / TCP)

The *same* system recurs — a client-server app reachable over HTTP, WebSocket, UDP, and TCP — across
languages ([gallows-remastered](https://github.com/Andrew1407/gallows-remastered),
[ClientConnectionStrategies](https://github.com/Andrew1407/ClientConnectionStrategies) + its UE plugin +
Node [TestServersApp](https://github.com/Andrew1407/ClientConnectionStrategies/tree/main/TestServersApp),
[GallowsGame](https://github.com/Andrew1407/GallowsGame),
[DichBoxMobile](https://github.com/Andrew1407/DichBoxMobile) HTTP-only, RL training servers).

Shared design across all of them:
- One transport-agnostic interface; one concrete class/module per protocol; one factory by label.
- **Adapter pattern to normalize protocols**: TCP/UDP sockets are wrapped to look like a
  WebSocket-style `{ onmessage, send, close }` object so the higher-level connection/stage logic is
  written once (JS `makeSocketWrapper` / `SocketInterfaceAdapter`; same intent in C#/UE clients).
- A single config "bag" struct carries `host`/`port`/timeout/response-callback
  (`FClientOptions` in UE, `FClientOptions`-style structs, `config()` objects in JS).

If a task is "talk to a server," reach for this layered transport abstraction rather than calling
the protocol API directly at the call site.

---

## 3. Dispatch tables over switch / if-else chains

In every language, replace branching with data structures keyed by label:

- JS: lookup-table reducers (`makeReducer({initial, actions})`), `stages[label]`, `events[event]`.
- Unity C#: `Dictionary<WaveState, Action>` handler maps, `Dictionary<string, Func<...>>` factories.
- Go: `map[string]http.HandlerFunc{...}` route table, then `range`.
- Python: `dict(fifo=..., edf=...)` algorithm dispatch; and `match`/`case` for type dispatch.
- Kotlin: `when` with enum dispatch, `companion object` `defineType()` factories.

**Rule:** map input → handler via a dictionary/lookup, iterate or index it; only fall back to
`if/elif` when the conditions aren't enumerable.

---

## 4. Strict layer separation; pure logic isolated from IO / framework

Every project separates concerns by **folder**, and keeps the framework/IO at the edges:

- Kotlin: `api/` (network), `datatypes/` (DTOs), `mv/` (logic), `view/` (Fragments only), `tools/`.
- Unity: MonoBehaviours are thin lifecycle shells; *all* logic lives in injected pure C# classes.
- JS/TS: routes → controllers → DB/connector → storage managers; or NestJS vertical slices.
- Python: `containers.py`/`utils.py`/`main.py` per package; `console_output.py`/`file_output.py`/
  `plots.py` are the *only* files allowed to print or touch the filesystem.
- C++: interface/abstract base in the parent folder, each concrete impl in its own subfolder.

**Rule:** generated code should put pure logic in framework-free units and confine IO, UI, and
engine/framework calls to thin adapters. A "feature" gets its own folder; multiple implementations
of one contract go in a `types/` (or `Strategies/`, `Interfaces/`) subfolder.

---

## 5. Naming: role-encoding suffixes (consistent across languages)

Class/type names end in a suffix that states the role. The same vocabulary appears in C#, C++,
Kotlin, TS, Python:

`*Strategy`, `*Factory`, `*Manager` (owns a pool/collection), `*Controller` (orchestrates one
domain), `*Service`, `*Handler`, `*Container` (holds data/values), `*Selector`, `*Builder`,
`*Adapter`, `*Observer`, `*Verifier`, `*Profiler`, `*Spawner`, `*Params` (config bag),
`*System` (a component doing complex logic), `*Utils`/`*Tools` (stateless helpers).

Other cross-language naming constants:
- **Enums:** `SCREAMING_SNAKE_CASE` members in *every* language (incl. C# and Kotlin where PascalCase
  is conventional) — e.g. `WaveState.ALL_WAVES_SURVIVED`, `ErrorStatus.SESSION_NOT_EXISTS`,
  `Privilege.PRIVILEGED_USER`, `TokenType` members.
- **Interfaces:** isolated in an `Interfaces/` folder or an `interfaces.go`/`I*.ts` file; the `I`
  prefix is used in C++/C#/TS (and is part of the UFUNCTION `Category` string in UE).
- **Boolean variables/methods:** read as predicates — `is*`/`was*`/`should*`/`able*`/`has*`
  (`shouldMove`, `wasSwapped`, `ableToPlay`, `passwdExpired`).
- **Constants:** `SCREAMING_SNAKE_CASE` for config/limits in all languages.
- **Private helpers** lean on verb prefixes: `make*`, `get*`, `format*`, `parse*`, `check*`,
  `build*`, `install*`/`bind*` (DI).

---

## 6. Static "named constructor" factory methods

Instead of exposing raw constructors, add a static factory — most distinctively a short
`of(...)` / `Of(...)`:

- JS/TS: `DominoTile.of(left, right)`, `GameSessionError.notExists()`, `ClientDB.getInstance()`.
- Python: `Token.of(...)`, `DynamicConveyor.of(...)`.
- Unity C#: `DamageModifierPredictor.FromModel(model)`, UDP `Strategy.Of(uri)`.
- Go: `New*` constructors universally; `CreateServer(...)` returning an interface.
- Kotlin: `*.of`-style and `companion object` factories.

Prefer a named factory (`of`, `from*`, `make*`, `New*`, `Create*`) over a bare `new`/constructor
at call sites, especially for value objects and error objects.

---

## 7. Tuple / pair returns to avoid multiple return paths

He returns small tuples/pairs rather than out-params or many code paths:

- JS/TS: `responseTuple = [statusCode, body]`; `getMoveActionParams(): [boolean, DominoTile]`.
- Kotlin: every API call returns `Pair<Int, C>` (HTTP status + parsed body), destructured at call site.
- Python: `build_model()` returns `(model, mse)`; type aliases like `FramePair = tuple[DataFrame, DataFrame]`.
- C++ / Kotlin algorithms: timing wrappers return `(elapsedTime, result)`.
- Go: named returns `(closed bool)` for loop-termination signals.

Destructuring the returned tuple immediately at the call site is the norm.

---

## 8. Timing/benchmark wrapper returning elapsed time

Algorithmic and parallel code wraps the work in a function that returns elapsed time as a number,
and compares sequential vs parallel variants side by side in `main`:

- C++: `double runCycle(...) { start = omp_get_wtime(); ...; return omp_get_wtime() - start; }`;
  `multiply` vs `multiplyParallel`, `optimized` vs `unoptimized`.
- Kotlin labs: `findSolution(): Pair<Long, Result>` where first element is elapsed ms.

Generate benchmarks as "wrap → return elapsed → compare named variants," not ad-hoc timestamps.

---

## 9. Config & lifecycle conventions

- **Env/config with inline defaults**: `process.env.PORT || 8080`, `os.getenv(...)`, default-valued
  flags. A `config()` factory function (JS/Python) or a constants block at module top (Python,
  C++ anonymous namespace, Go `var` flag block).
- **Entry point is minimal**: `main`/`if __name__ == '__main__'` just calls a `make_*()`/factory and
  one lifecycle method (`launch()`, `run()`, `main()`). All wiring lives in the factory.
- **Graceful shutdown** is implemented deliberately in every server (JS `shutdownOnce()` +
  force-exit `setTimeout(...).unref()` on SIGINT/SIGTERM; Go signal goroutine / `signal.WaitFor...`;
  Python `except (KeyboardInterrupt, EOFError)`).

---

## 10. Immutability & defensive copies

- Accessors return **copies**, not internal references: Python `get_*()` returns `tuple(...)`;
  JS value objects expose `copy()`/`copyReversed()` and have no setters.
- Immutable constant maps: JS `Object.freeze({...})`; Python class-as-namespace constants;
  enums everywhere instead of magic values.
- Python initializes empty collections with `list()` (not `[]`) and returns `tuple(...)`.

---

## 11. Testing conventions (cross-language)

- **Hand-written mocks/stubs** implementing the interface — no mocking frameworks (Unity `WorkerMock`,
  UE test-client subclasses, Kotlin none, Go `testify`/manual).
- **Descriptive test names**: `Should_Reset_Modifications_To_Default` (C#), `test_nesting_valid`
  (Python), `verifiesOmittingCorrect` (Kotlin).
- **Data-collection-then-iterate** instead of many test functions: build a `tests` object/map of
  name→fn and loop it (JS `for (const k in tests) it(k, tests[k])`, Go map-keyed tables).
- **Test folder mirrors source folder** (Unity `Tests/EditMode/...`, Python `test/`, parallel `__tests__/`).
- **Parameterized cases**: `it.each([...])` (JS), `t.Run(...)` lifecycle subtests (Go).
- Custom lightweight runners when the framework doesn't fit (DichBox `ITester` with `test()`/`run()`).

---

## 12. Small recurring idioms

- **Guard clauses / early return** everywhere; no deep nesting (`if (!valid) return;`).
- **`#pragma region` / `#region`** to group members inside large classes (C++, UE, C#); region names
  are `UPPER_SNAKE_CASE` (C++/UE) or `PascalCase` (C#).
- **2-space indentation** is the modern-code default (Python — distinctively non-PEP8, modern C++, JS/TS).
  Some older code uses tabs; use 2-space for new code.
- **Higher-order factory functions** to parameterize behavior (`make*({...})`, currying via
  `bind`/`partial`/`WithArguments`).
- **Spelling quirks are preserved** across refactors (e.g. `NeuralNetrwork/`, `Modicitaions`,
  `ParsingExeprion`, `IResetable`); when extending existing code, match the established (mis)spelling of
  an identifier rather than "fixing" it.

---

## Per-language deep dives

- [JavaScript / TypeScript](./javascript-typescript.md)
- [C++ (general)](./cpp.md)
- [Unreal Engine 5 C++](./unreal-cpp.md)
- [Unity C#](./unity-csharp.md)
- [Go](./go.md)
- [Python](./python.md)
- [Kotlin](./kotlin.md)
