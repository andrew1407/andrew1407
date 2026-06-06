# Kotlin Style

Repos: [DichBoxMobile](https://github.com/Andrew1407/DichBoxMobile) (Android, Retrofit, HTTP client for the DichBox
API — the architectural reference), [EmbeddedSystems](https://github.com/Andrew1407/EmbeddedSystems) Kotlin labs
(Ferma function, perceptron, genetic algorithm).

Read [common.md](./common.md) first.

---

## Package taxonomy

- Root package `<org>.<appname>`; first-level sub-packages are strictly categorical (camelCase, singular):
  - `api/` (network; sub-packages mirror REST resources: `api/user`, `api/boxes`)
  - `datatypes/` (ALL sealed-class DTO hierarchies — not "models"/"data")
  - `mv/` (all non-Fragment/Activity logic: presenters, verifiers, view-model states; grouped by entity)
  - `view/` (**Fragment subclasses only**)
  - `tools/` (package-level free functions + stateless enums)
- **Zero mixing**: Fragments only in `view/`, logic only in `mv/`, network only in `api/`, DTOs only in `datatypes/`.
- Algorithmic labs: sub-package named after the algorithm (`perceptron/`, `geneticAlgorithm/`).

## Naming suffixes

`API` (concrete Retrofit client), `Service` (Retrofit `interface`), `Container` (sealed DTO holder),
`ViewModel`, `Presenter`, `Profiler` (fills views via fluent chain), `Handler`, `Fetcher`, `Verifier`,
`Adapter`, `Dialog`, `Editor`, `Fields` (enum of form keys), `Redirector`/`Cleaner` (callback interfaces).
Suffix-less files = top-level functions (`ContainerParsers.kt`, `ViewDecorators.kt`, `ImageConverters.kt`).

## Retrofit three-layer architecture (strict)

1. **`*Service` interface** — thin: every endpoint `@POST`, body raw `RequestBody`, return
   `Response<ResponseBody>`, `suspend fun`. No deserialization here.
2. **`ApiParser<C>` abstract base** — owns the Retrofit instance (hardcoded baseUrl); implements
   `makeRequest(entries): RequestBody` and `getResponseData(response, respClass): Pair<Int, C>`; declares
   abstract `parseJSON`/`stringifyJSON`.
3. **`*API` concrete** — `extends ApiParser<Container>`, creates the service, one `suspend fun` per endpoint,
   each a three-line body `makeRequest → service.call → getResponseData`; overrides parse/stringify by
   delegating to the sealed class companion.

Every call returns **`Pair<Int, C>`** (HTTP status + parsed body), destructured at the call site; status
checked via `Statuses.OK.eq(st)` / `.eqNot(st)`.

## Sealed-class DTOs

- All DTOs for a domain are **inner `data class`es of one `sealed class`** (`UserContainer`, `BoxesContainer`);
  the sealed class holds only a `companion object` delegating `parseJSON`/`stringifyJSON` to package-level
  Gson helpers.
- Request + response types live in the same sealed class, differentiated by name (`VerifyField`/`VerifyFieldRes`).
- Server-mirroring fields in **snake_case** (`name_color`, `user_uid`); app-coined fields camelCase. Optional
  fields `val x: T? = null`.

## ViewModel template

```kotlin
class XViewModel : ViewModel(), Cleanable {
  private val data = MutableLiveData<T>()
  val liveData: LiveData<T> = data
  fun setX(v: T?) { data.value = v }
  override fun clear() { if (data.value != null) data.value = null }
}
```
Backing `data` always private; exposed property always `val`; `Cleanable` (single `clear()`) always implemented;
complex VMs reassign `data.value` immutably rather than mutating.

## Async

- **Fire-and-observe** for UI actions: `CoroutineScope(Dispatchers.Main).launch { ... withContext(Dispatchers.IO){ api.call() } }`
  inline (no stored scope).
- **`runBlocking`** for init-time fetches before UI is ready.
- All Service + API methods `suspend fun`; result `Pair` destructured immediately; labeled returns
  (`return@launch`, `return@observe`); never throws.

## Null-safety

- **No scope functions at all** — no `let`/`run`/`also`/`apply`/`with` anywhere, even for builder setup.
- Explicit `if (x == null) return`, `!!` where provably non-null, Elvis with `return`/default,
  nullable destructuring `val (name, _) = res as UserContainer.SignedContainer`.

## Fluent builder chaining (dominant composition idiom)

Handler/Verifier/Profiler setup methods return `this`:
```kotlin
SignUpVerifier(submitBtn)
  .checkUsername(username, nameWarning)
  .checkEmail(email, emailWarning)
  .checkPassword(password, passwordWarning)
```
`init {}` blocks wire constructor-time setup (e.g. attaching a click listener to the `Button` ctor arg);
every later configuration step is chainable.

## Other conventions

- View refs & deps are `private lateinit var`, assigned in `onViewCreated`; lazy guards via
  `this::field.isInitialized`.
- **Enums with behavior**: a `val` property + helper methods (`Statuses.OK.eq(st)`, `FieldsTemplates.NAME.test(s)`),
  `companion object` dispatch `when` (`NotificationsTypes.defineType`), abstract method per entry with overrides,
  `getVal() = name.toLowerCase()` to bridge to server snake_case.
- **No DI framework** — manual constructor wiring; `mv/` classes hold `private val api = UserAPI()`; ViewModels
  always activity-scoped (`ViewModelProvider(requireActivity())`).
- Thin single-method callback interfaces (`FragmentsRedirector`, `Cleanable`, `FragmentCleaner`) cast from the
  Activity. `tools/` is free functions + a stateless enum; `internal fun` Gson one-liners in `ContainerParsers.kt`.
- `Pair` destructuring everywhere; `when` as statement/expression with sentinel error codes (`-1L`, `-2L`);
  `SpannableString` built manually; `AppColors` enum for all colors (no inline hex); early-return guard clauses.

## Algorithmic code (EmbeddedSystems)

- One class per algorithm, ctor takes all config; entry point `findSolution()`/`calcWeights(...)` returns
  `Pair<Long, Result>` (elapsed ms first). Algorithm in its own sub-package; data types as separate
  `data class` files. Stateless function-level algorithms are top-level functions named `PascalCase`
  (`FermaFunction(...)`). A thin `InputHandler` glue class takes a `Button`, wires the listener in `init`,
  then uses the same fluent-chain setters.

## Tests

- Unit tests only for logic classes (`InputVerifier`). `<Class>Test` naming, single descriptive `camelCase`
  method (`verifiesOmittingCorrect`), JUnit4 (`assertEquals`/`assertThrows`), `runBlocking {}` for suspend,
  **no mocking frameworks**. androidTest has only the default scaffold.
