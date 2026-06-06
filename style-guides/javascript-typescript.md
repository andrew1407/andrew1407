# JavaScript / TypeScript Style

Repos: [Domino](https://github.com/Andrew1407/Domino) (NestJS + Next.js/React, TS, Redis, WS),
[DichBox](https://github.com/Andrew1407/DichBox) (Express + React, TS, PostgreSQL),
[gallows-remastered](https://github.com/Andrew1407/gallows-remastered) (Node, console+browser clients, Redis, HTTP/WS/UDP/TCP),
[ClientConnectionStrategies/TestServersApp](https://github.com/Andrew1407/ClientConnectionStrategies/tree/main/TestServersApp) (Node multi-protocol server).

Read [common.md](./common.md) first — the Interface→Strategy→Factory triad, dispatch tables,
tuple returns, and multi-protocol architecture below are cross-language traits.

---

## Project layout

- Always split into independent `client/` and `server/` apps, **each with its own `package.json`**.
  Never a workspace/monorepo root package.
- Server root: flat `index.ts`/`index.js` entry + domain modules as sibling directories.
- gallows multi-protocol layout (canonical):
  ```
  strategies/<strategyName>/         # e.g. dataraw/, streamable/
    services/   http.js ws.js tcp.js udp.js
    clients/browser/  http.js ws.js
    clients/console/  http.js ws.js tcp.js udp.js
    connection.js events.js loader.js stages.js
  ```

## File / folder naming

- TS interfaces: `I`-prefixed file **and** type — `ILogger.ts` → `interface ILogger`. (One exception:
  `StorageClient` interface in Domino has no `I`.)
- NestJS layer files: `*.service.ts`, `*.gateway.ts`, `*.controller.ts`, `*.module.ts`. DI tokens in
  `*.options.ts`. Outside Nest: plain PascalCase (`Logger.ts`, `ClientDB.ts`).
- Grouping folders: camelCase (`gameSession/`, `wsTools/`, `storageManagers/`, `controllersFactory/`).
- React components: PascalCase `.jsx` + matching `.module.scss`. Bare Node modules: camelCase
  (`sessionEmitter.js`, `socketConnection.js`, `playerFormatters.js`).

## TypeScript conventions

- **Annotate every variable**, even trivial: `const delay: number = 1000`. Explicit return types on
  **all** methods (`: Promise<void>`).
- `const enum` (never plain `enum`), own file, default-exported, `SCREAMING_SNAKE_CASE` members:
  ```ts
  const enum MoveState { AVAILABLE, SKIPPABLE, DEAD_END }
  export default MoveState;
  ```
- Types co-located with their consumer (inline above/inside the class); shared domain types in
  `datatypes.ts` or the entity file. `type` for unions/tuples/aliases, `interface` for extendable shapes.
- `as` only to narrow after validation; `Partial<T>` for expected objects in tests.
- `public` always explicit; injected deps `private readonly`.

## Classes & abstractions

- Private state: native `#field` in JS classes; `private readonly` in TS.
- Singleton via private constructor + static `getInstance()` (DichBox `ClientDB`).
- Value objects: static `of(...)`, immutable, `copy()`/`copyReversed()`, no setters (`DominoTile`).
- Abstract base classes with `protected` shared logic + required subclass methods (`Validator`,
  `StorageManager`).
- DichBox factories (`BoxesControllerFactory implements IControllerFactory`) assemble the layer stack
  and are instantiated once at module load.

## Error handling

- **Static error factory class** (`GameSessionError.notExists()`, `.badRequest()`, `.internal()`),
  plus a `catchHandler()` static returning a **method decorator** applied above every `@SubscribeMessage`
  handler. Unknown errors → delayed `gatewayShutdown` (fail-fast).
- **`responseTuple` pattern** (DichBox): handlers return `[statusCode, body]` via `makeTuple(st, obj)`;
  a `getWrappedRoutes()` HOF turns handler objects into Express middleware. Business errors never throw.
- Guard clauses + early return; match specific error message strings when no custom class exists.

## Async

- `Promise.all([...])` for independent ops, destructuring the tuple of results.
- `setTimeout` from `'timers/promises'` for awaitable delays; top-level `await` in ESM entries.
- Fire-and-forget long tasks with `.catch(handler)` (no `await`).
- `Promise.allSettled` for best-effort cleanup.

## Functional idioms

- `Object.freeze({...})` for constant maps; **lookup tables over switch** (`stages[label]`,
  `events[event]`, `getters[type]`).
- HOF factories: `makeServiceDescriptor({service, connectionAdapter})`,
  `makeConnectionHandler(...)`, `makeFormatter(formatters)`, `makeReducer({initial, actions})`.
- `bind` for partial application (`dispatchObj.bind(null, TYPE)`).
- `for...in` over objects / `for...of` over arrays for side-effects; `??=` lazy init; `?.`/`??` throughout.
- Numeric separators (`26_000_000`), `parseInt(x, 10)` always with radix.

## Multi-protocol architecture

- `strategiesTooling.js`: `connections = Object.freeze({ws, http, udp, tcp})`, `strategies = {...}`.
- Server entry dynamically imports the chosen `strategies/<name>/services/<protocol>.js` at runtime.
- **Adapter** normalizes TCP/UDP sockets to a WS-like `{onmessage, send, close}` so connection/stage
  logic is protocol-agnostic (`makeSocketWrapper`, class `SocketInterfaceAdapter` with `#fields`).
- TestServersApp: each `servers/<protocol>.js` default-exports `createServer(port, host, onData)`.

## State / sessions

- Redis keys namespaced via `prefixedNamespace(...args)` → `'domino:player_deck:<id>:<player>'`.
- In-memory gateway sessions as a private `Record<sessionId, SessionPlayer[]>` field.
- Symmetric serialization with `makeFormatter` (`toStorageFormat`/`fromStorageFormat`).

## React / Redux (clients)

- Reducers as **lookup tables**, not `switch`: `makeReducer({initial, actions})`; action types as
  `SCREAMING_SNAKE_CASE` constants in `storage/types.js`; action creators via `dispatchObj.bind(null, TYPE)`.
- Emitter + Listener class pair for WS (`SessionEmitter` sends, `SessionListener` dispatches);
  `socketConnection()` factory wires both and returns the emitter.
- Components are flat/declarative, logic pushed into the emitter/listener/context layer.

## Tests

- NestJS: `*.spec.ts` in `__tests__/`; one `describe` per class/method; stubs as
  `Record<string, jest.Mock>`; `.overrideProvider().useValue()`; private state via `gateway['sessions']`;
  `it.each([...])` parameterization; `async (): Promise<void>` callbacks spelled out.
- DichBox: custom `ITester` runner (`test()`/`run()`), sequential test classes; React tests build a
  `tests` object then `for (const k in tests) it(k, tests[k])`.
- TestServersApp: built-in `node:test`/`node:assert`, servers started inline, raced against a timeout.

## Config & shutdown

- `dotenv` + a `config()` factory for `ConfigModule.forRoot({load:[config]})`; JSON config via
  `import x from './env.json' assert {type:'json'}`. Env vars always have `||`/`??` defaults.
- Graceful shutdown: `shutdownOnce()` closure, close all sessions, force-exit
  `setTimeout(process.exit, delay, 1).unref()`, on SIGINT/SIGTERM.
