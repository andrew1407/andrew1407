# Go Style

Repos: [G-V-G](https://github.com/G-V-G)/{[l1](https://github.com/G-V-G/l1), [l3](https://github.com/G-V-G/l3),
[l4](https://github.com/G-V-G/l4), [2.l1](https://github.com/G-V-G/2.l1), [2.l2](https://github.com/G-V-G/2.l2)} —
group projects. Authorship is
**not** attributable per file; these are reported as team conventions, with notes where a pattern appears
only in repos Andrew definitely touched (l1, l3, 2.l2).

Read [common.md](./common.md) first.

---

## Layout

- `cmd/<name>/` for every binary; shared domain code in flat top-level packages. **No `internal/`, no
  `pkg/`**, no nesting beyond two levels.
- A package is **split into multiple files by concern**, not one big file: e.g. `handlers/` =
  `httpHandlers.go` (dispatcher methods) + `forumsComponents.go`/`usersComponents.go` (verb helpers) +
  `newHandler.go` (constructor only); `datastore/` split into `db.go`, `entry.go`, `merge-handler.go`,
  `write-handler.go`, etc.; `engine/` has a dedicated `interfaces.go`.
- File names camelCase **or** hyphen-case, never snake_case (inconsistent — both acceptable).

## Naming

- Packages: short lowercase (one camelCase outlier `generalStore`).
- Constructors always `New<Type>`; framework factories `<Name>Factory`.
- HTTP handlers on a `Handlers` struct: `Handle<Entity>` for multi-method dispatch, `Get<Entity>` for reads;
  unexported per-verb helpers (`addForum`, `getUsers`) in separate files.
- Receivers: short abbreviations (`db`, `fs`, `gs`, `h`, `e`, `q`, `mh`, `wh`).
- Named map/slice types for domain concepts: `type hashIndex map[string]int64`, `type Report map[string][]string`,
  `type HostsHealth []server`.

## Structs & interfaces

- Interfaces in a dedicated `interfaces.go` (two+ related interfaces per file).
- **Exported interface + unexported impl + exported `Create*`/`New*` factory returning the interface**:
  ```go
  type Server interface{ Start() }

  type server struct{ httpServer *http.Server }

  func (s server) Start() { /* ... */ }

  func CreateServer(port int, handler http.Handler) Server {
      return server{httpServer: &http.Server{ /* timeouts, MaxHeaderBytes */ }}
  }
  ```
- Struct literals: named fields in multi-line; positional only for single-field wrapper structs
  (`&tools.Users{fullUsers}`).
- Embed `sync.Mutex` directly into structs (`Engine`, `Queue`).
- Named function types implementing an interface (`type onFinishFn func(Handler)` with an `Execute` method).

## Error handling

- **Signature pattern: `if val, err := ...; err != nil { ... } else { happy path }`** — the happy path
  goes in `else`, not after the guard. Recurs across commits, handlers, balancer, db.
  ```go
  if commitsJSON, err := json.MarshalIndent(&commits, "", " "); err != nil {
      rw.Write([]byte("{}"))
  } else {
      rw.Write(commitsJSON) // happy path lives in else
  }
  ```
- Early-return guard for method/precondition checks.
- `fmt.Errorf` for all errors, **no `%w` wrapping**, no `errors.Is`/`As` (pre-1.13 style). Sentinel errors
  in a `var (...)` block. Sentence-case messages, no trailing period.
- `panic(err)` for unrecoverable startup (even followed by `else`), `log.Fatalf` for init errors in `main`.
- Avoid the verbose `var textError string; ... ; fmt.Errorf(textError)` intermediate.

## Concurrency

- **Event-loop decomposition** (l4): `Command`/`Handler` interfaces, `Engine` embeds `sync.Mutex` + `*Queue`,
  loop in a goroutine; `Queue` blocks `Pull` on an empty-channel condition; terminal command sets a stop flag.
- **Handler-actor pattern** (2.l2 datastore): exported `Req`/`Res` channels + unexported `closed chan bool`;
  `StartLoop()` runs `for { if closed { break } }` in a goroutine then signals; `Close()` closes Req/Res
  and drains `<-closed`. Callback returns `bool` to signal closure; named return `(closed bool)`.
  ```go
  func (wh *WriteHandler) StartLoop() {
      go func() {
          for {
              if closed := wh.onWriteClb(); closed { break }
          }
          wh.closed <- true
      }()
  }

  func (wh *WriteHandler) Close() {
      close(wh.Req)
      close(wh.Res)
      <-wh.closed // drain to await termination
  }
  ```
- Health-check loops: `for i := range pool { i := i; go func(){ for range time.Tick(...) {...} }() }`.
- Graceful shutdown: signal goroutine on `os.Interrupt`; in 2.l2 extracted to `signal.WaitForTerminationSignal()`.

## HTTP

- Mature style: routes as `map[string]http.HandlerFunc{...}` then `range` to register.
- `Handlers` struct holds the store pointer; method routing via `if/else if` on `req.Method`; params named
  `rw http.ResponseWriter, req *http.Request`.
- Server construction encapsulated in `CreateServer(port, handler) Server` with configured timeouts.
- JSON response helpers `WriteJsonOk`/`WriteJsonBadRequest`/`WriteJsonInternalError` wrapping a private
  `writeJson`; `content-type: application/json`.

## DI & data

- `google/wire` for production servers (`build.go` + committed `wire_gen.go`, `providers = wire.NewSet(...)`,
  short import aliases). Manual wiring for smaller services.
- Wrapper structs for arrays (`type Users struct{ UsersArr []*User \`json:"users"\` }`) so metadata can be
  added later. `make([]T, 0)` for empty slices (not `[]T{}`). Domain layer declares zero-value `var`s at the
  top before logic.

## Tests

- **No table-driven tests.** Sequential narrative assertions. Two styles: `testify/assert` and
  `t.Errorf`/`reflect.DeepEqual`. `t.Run` for lifecycle phases (not data sets). Package-level `var` fixtures.
  Local helper closures inside test functions. Occasionally map-keyed combinatorial tables (`map[[3]bool][]string`).

## Quirks — do NOT reproduce

Parenthesized `if (err != nil)`, space before param lists `func Foo (...)`, mixed-space indentation
(un-gofmt'd files), `for i, _ := range`. Use standard tabs + `gofmt`.
