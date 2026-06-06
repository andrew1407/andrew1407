# C++ (general) Style

Repos: [ComputerScienceEssentials](https://github.com/Andrew1407/ComputerScienceEssentials) (data structures, sorting, patterns),
[OOP](https://github.com/Andrew1407/OOP), [SecuritySystems](https://github.com/Andrew1407/SecuritySystems)
(crypto, disk model, layered access), [NumericalTechniques](https://github.com/Andrew1407/NumericalTechniques),
[OperatingSystems](https://github.com/Andrew1407/OperatingSystems) (allocators, cache),
[ParallelOpenMP](https://github.com/Andrew1407/ParallelOpenMP) (OpenMP/MPI),
[ParallelCuda](https://github.com/Andrew1407/ParallelCuda) (CUDA). Unreal C++ is a **separate** file
([unreal-cpp.md](./unreal-cpp.md)) — do not mix the conventions.

Read [common.md](./common.md) first.

---

## Formatting & file structure

- **2-space indent** in mature projects (SecuritySystems, CSE, OS labs 1/2/5/6) — use this for new code.
  (Older repos use tabs; don't reproduce.)
- **`#pragma once`** always — never `#ifndef` guards.
- Modern projects: headers `.hpp`, impl `.cpp`. WinAPI/OOP context: `.h`. CUDA: `kernel.cu` + `.h`.
- File naming: **PascalCase for class files** (`AccessController.cpp`); **camelCase for
  namespace/utility files** (`dataparser.hpp`, `functionCheck.cpp`); lowercase for algorithm files
  (`bubble.cpp`); data-structure files get a `Self` suffix (`stackSelf.hpp`); labs are
  `N-lab.cpp`, `N-lab.test.cpp`, `N-lab.threads.cpp`.

## Project organization

- Each conceptual module gets its **own directory** (not just a file). The interface/abstract base
  sits in the parent folder; each concrete implementation in a subfolder named after itself.
  Third-party code isolated under `lib/`.
  ```
  accessController/layers/
    Layer.hpp                  # pure interface in parent
    diskReader/DiskReader.hpp/.cpp
    loginForm/passwordInputLog/PasswordInputLog.hpp/.cpp
  ```
- Build: **shell scripts only** (`run.sh`), glob all `.cpp` with `**`, output to `out/` or `app`,
  run via `$_`. No CMake/Makefile.

## Namespaces

- Namespaces group **free-function modules**, not classes: `namespace sorting`, `self`, `utils`,
  `dataparser`, `crypto`, `cmd`, `logger`, `functionCheck`, `fft`.
- Classes live at file/module scope, **not** inside namespaces.
- `using namespace std;` in `.cpp` files only — never in headers.
- File-local helpers: forward-declare at top of `.cpp`, define after the namespace block.
- Module-local constants and internal helpers go in an **anonymous `namespace { ... }`** at the top of
  the `.cpp` to give them internal linkage:
  ```cpp
  namespace {
    constexpr uint PASSWD_LEN_MIN = 4;
    bool isAllowed(const std::string& path);
  }
  ```

## Class design

- Layout order **`public:` then `private:`** (exception: implementation-heavy classes like `Allocator`
  put `private:` first).
- Abstract bases are minimal — pure virtuals only, no data (`class Layer { virtual bool run() = 0; ... }`).
- One-liner trivial bodies inline in the header; longer methods in `.cpp`.
- `this->` used when a method touches multiple members or member/param shadowing is possible.

## Naming

- Classes PascalCase with role suffixes (`*Strategy`, `*Factory`, `*Builder`, `*Adapter`, `*Visitor`,
  `*Container`).
- Members: **`_` prefix for low-level implementation classes** (allocators, data-structure internals);
  **no prefix** for domain-logic classes.
- Methods/vars/functions camelCase; booleans `is*`/`was*`/`should*` (plus the personal `breakpoint`
  bool as a while-loop sentinel).
- Constants — `#define`, `constexpr`, and `const` globals all `SCREAMING_SNAKE_CASE`.
- **Unscoped `enum`** (never `enum class`) with `SCREAMING_SNAKE_CASE` values; accessed `Shape::CIRCLE`.

## Types & memory

- **`typedef`, never `using`** for aliases; `typedef struct {...} Name;` inside namespaces for POD.
- **Raw pointers with manual `new`/`delete`** throughout — no `unique_ptr`/`shared_ptr` (one `popen`
  exception). Owning pointer members: init `nullptr` in ctor list, `delete` with null-check in dtor.
  `new (std::nothrow)` for null-checkable allocation.
- Aggressive `const` on computed locals in parallel/numerical code; `constexpr` for modern constants.

## Templates

- Templates for **generic data structures and simple search only** — no metaprogramming/SFINAE.
- Keep template definition in `.cpp` with **explicit instantiation** at end of header
  (`template class Stack<int>;`).

## Design patterns (CSE `utils/patternsUsage/`)

- One flat `.cpp` per pattern; pattern classes file-scoped; only the `utils::useFoo()` runner exported.
- **Factory/Builder by rvalue-ref**: a use-function takes the factory by `&&` and is called with a
  temporary concrete factory:
  ```cpp
  void useStorage(StorageFactory&& factory) {
    Storage* storage = factory.create();
    // ...
    delete storage;
  }
  useStorage(OnlineStorageFactory()); // concrete factory passed as a temporary
  ```
- Strategy: abstract base + `makeShape(Shape, ...)` returning raw pointer.
- Observer: `std::multimap<const char*, Subscriber*>` + `equal_range`.
- Decorator: template abstract base `DataContainer<T>` holding `wrapee` by reference.
- Singleton: non-Meyer's — static raw pointer, deleted copy ctor/assign.

## Algorithm decomposition

- Extract timing into a wrapper returning elapsed `double` (`calcTime`, `runCycle`), then compare named
  sequential vs parallel variants (`multiply`/`multiplyParallel`) in `main`:
  ```cpp
  double runCycle(const std::function<void()>& work) {
    const double start = omp_get_wtime();
    work();
    return omp_get_wtime() - start; // elapsed seconds
  }
  const double seq = runCycle(multiply);
  const double par = runCycle(multiplyParallel);
  ```
- `bool breakpoint` sentinel instead of `while(true){break;}`.
- `main` = generate input → process → verify → cleanup, in labeled blocks.
- `printf` for numerical output; `std::cout`/`cin` for domain IO.

## OpenMP / MPI

- `omp_set_num_threads(N)` at top of `main`; reduction with named accumulators declared before the
  pragma; `#pragma omp task if(...)` gated by a member flag; atomic/reduction over locks.
- MPI: `MPI_Init`/`MPI_Finalize` bracket, `MPI_Wtime()`, rank variable named `worldRank`.

## CUDA

- Every function annotated `__global__`/`__device__`/`__host__`; `__host__` `execX()` wraps dispatch+timing.
- Thread index `const int i = threadIdx.x + blockIdx.x * blockDim.x;`.
- Host pointers suffixed `h`, device pointers `d`; `cudaMalloc`/`cudaFree` paired; explicit memcpy direction.
- `cudaEvent_t` timing; `printf` inside `__host__` display functions; `std::function` typedef for lambda iterators.

## Avoid

Do not use `enum class`, `using` aliases (use `typedef`), range-for over raw arrays, C++17 attributes
(`[[nodiscard]]` etc.), Doxygen, header-only templates, or CMake/Make (shell scripts only).
