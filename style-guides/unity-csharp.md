# Unity C# Style

Repos: [Reactivation](https://github.com/Andrew1407/Reactivation) (3D TPS, Zenject, Barracuda/ONNX, waves),
[GallowsGame](https://github.com/Andrew1407/GallowsGame) (2D, Zenject, HTTP/WS/UDP/TCP).
All logic under `Assets/Scripts/`.

Read [common.md](./common.md) first.

---

## Folder taxonomy & namespaces

- **No `namespace` declarations** anywhere — every type is global. Folders are the only organizing axis.
- Top-level folders by role: `Installers/` (with `Global/`, per-prefab subfolders), `MonoBehaviours/`,
  domain folders (`Player/`, `Enemy/`, `Weapons/`, `Gameplay/`, `Stats/`, `Ui/`), `Observers/`,
  `Common/{DataStructs}`, `InputSystem/`.
- A `types/` subfolder holds concrete implementations of an interface; a `base/` subfolder holds abstract
  bases. GallowsGame mirrors this (`Gameplay/Online/` = one file per transport, `ViewControllers/`).

## MonoBehaviour vs pure C# (the defining rule)

- **MonoBehaviours are thin lifecycle shells**: only `[SerializeField]` wiring, Unity lifecycle/physics
  callbacks, and coroutine hosting. They delegate immediately to injected pure C# services.
- **All stateful logic is plain C# classes** injected by Zenject (`GameManager`, `WavesController`,
  `PlayerHealthController`, `LocomotionStateMachine`, `AmmoManager`, networking clients, ML predictor).
- Pure C# classes never call `GetComponent` or use `Awake`/`Start` — components are extracted in the
  Installer or in an `[Inject] Construct()` method; lifecycle comes from `ITickable`/`IInitializable`.

## Zenject architecture

- **One `MonoInstaller` per subsystem**, never merged. Naming `[Subsystem][Context]Installer`.
- **Sub-container for private deps**: `BindInterfacesAndSelfTo<T>().FromSubContainerResolve().ByMethod(installT)`
  whenever a class has dependencies that shouldn't leak; simple classes get plain `Bind<T>().AsSingle()`.
- `BindInterfacesAndSelfTo` for anything implementing `ITickable`/`IInitializable`/`ILateDisposable`.
- `BindInstance(s)` for SerializeField structs/prefab refs; `WithArguments` for ctor primitives;
  `WithId("...")` + `[Inject(Id="...")]` to disambiguate same-type bindings; `WhenInjectedInto<T>()` for scoping.
- Prefab factories: nested `public class Factory : PlaceholderFactory<T> {}` inside the MonoBehaviour,
  bound `BindFactory<T, T.Factory>().FromSubContainerResolve().ByNewContextPrefab(prefab)`.
- **Events go through the custom Observer pattern** (below) rather than Zenject's `SignalBus`.
- `[Inject] private void Construct(...)` for field-injection when ctor injection won't fit or needs an Id.

## Naming

- Class suffixes: `Installer`, `Observer`, `Controller`, `Manager`, `Strategy`, `Factory`, `Spawner`,
  `Container`, `Selector`, `Params` (`[Serializable]` config struct), `State`, `Behaviour`, `Utils`/`Labels`.
- Fields: `_camelCasePrivate` (always, underscore prefix); public readonly `PascalCase`; const private
  `_camelCase`, const public `UPPER_CASE`.
- **Methods: public `PascalCase`, private `camelCase`** (distinctive); install helpers `install*`/`bind*`.
- Enums: `ALL_CAPS_SNAKE_CASE` values; `[Serializable]` struct fields `PascalCase` (no underscore).

## Observer / event pattern

- Wrap `UnityEvent` inside a dedicated pure-C# Observer class; expose a typed `On[Event](...)` method;
  subscribe a private listener in the constructor. Callers fire `On[Event]` without knowing subscribers.
  ```csharp
  private readonly UnityEvent<string,int> _onAmmoChange = new();
  public PlayerAmmoObserver() => _onAmmoChange.AddListener(changeUiInfo);
  public void OnAmmoChange(string l, int c) => _onAmmoChange.Invoke(l, c);
  private void changeUiInfo(string l, int c) => _uiController.SetAmmo(l, c);
  ```
- GallowsGame replaces this with `async/await` + `TaskCompletionSource` to bridge coroutines/UI to tasks.

## Data structures

- Groups of config values are **`[Serializable] struct`** surfaced as `[SerializeField]` on Installers
  (`MotionParams`, `AmmoParams`, `CharacterComponents`).
- Static-only utilities are **`public sealed class`** (never `static class`), sectioned with `#region`.

## Async style

- Coroutines (`IEnumerator`) are the main async tool in Reactivation; pure classes return them,
  MonoBehaviours `StartCoroutine`. "Run one action at a time" helper wraps + tracks `_currentAction`.
- GallowsGame networking is fully `async/await`; `TaskCompletionSource` bridges events; `async void` only
  on UI button callbacks.
- Input subscription centralized in `setInputActionsState(bool enabled)` called from `OnEnable`/`OnDisable`.

## Networking abstraction (GallowsGame)

- One `IGameplayStrategy { Task Setup(); Task<GameProgress> StageAction(string); }`; one
  `[Protocol]ClientStrategy` per transport (all `IDisposable`); `GameplayStrategyFactory` keyed by URI
  scheme; UDP uses static `Of(uri)`. Serialization: Newtonsoft with `[JsonProperty]` + `NullValueHandling.Ignore`.

## ML / Barracuda

- `DamageModifierPredictor` is pure C#, static `FromModel(NNModel)` factory, ctor takes `IWorker` (mockable),
  implements `IDamageModifier` + `IDisposable`; bound via `FromInstance(...FromModel(model))`.

## Recurring idioms

- His own `IResetable` (one `s`) with `ResetState()` across restartable classes/structs; cleanup via
  pattern-match: `if (x is IResetable r) r.ResetState(); if (x is IDisposable d) d.Dispose();`.
- Generic thread-safe `ObjectPool<T>` with custom `+`/`-` operators and `lock`.
- **Dictionary-dispatch** (`Dictionary<State, Action>`) instead of switch/if-chains — used pervasively.
- Cached animation hashes as `readonly int _xHash = Animator.StringToHash("x")`.
- `lock` on shared mutable state in cross-thread pure classes; `=>` expression bodies for short members;
  `[Header(...)]` grouping of `[SerializeField]`s; `[Inject] private readonly` fields.

## Tests

- NUnit + Unity Test Framework. `[Test]` EditMode (pure logic), `[UnityTest] IEnumerator` PlayMode
  (extends `ZenjectIntegrationTestFixture` with `PreInstall`/`PostInstall`).
- Test names `Should_[Behaviour]_[Condition]`. **Hand-written inner mock classes** implementing the
  interface (no mock framework). `Assert.*(..., message: $"...")` on every assertion. Test folder mirrors
  source. Helper factories `make[Subject](...)`. File-level `using Alias = Long.Generic.Type;`.

## Avoid

Do not use `namespace` declarations or `static class` (use `sealed class` for utilities). Keep lifecycle
in Zenject interfaces (`ITickable`/`IInitializable`) rather than `Awake`/`Start` in pure classes.
