---
name: my-code-style
description: Refactor, restructure, and prettify code so it follows Andrew Golovko's personal style guides (common architecture rules plus per-language rules for JS/TS, C++, Unreal C++, Unity C#, Go, Python, Kotlin). Use when asked to apply "my code style", clean up / restructure / prettify code, or make a project match these style guides.
---

# my-code-style

Refactor and restructure code in the current project so it matches the style guides listed
below. The guides describe the architecture, naming, decomposition, and formatting patterns
used across this author's repositories. Apply them only to languages the project actually
contains.

## Style guide sources

The guides live in one place only: the public GitHub repo `andrew1407/andrew1407`. Always
fetch them over the network — never assume a local checkout exists and never read them from a
filesystem path.

- Raw root (fetch from here): `https://raw.githubusercontent.com/andrew1407/andrew1407/main/style-guides/`
- Browsable root: `https://github.com/andrew1407/andrew1407/tree/main/style-guides`

To fetch a guide, append its file name to the raw root, e.g.
`https://raw.githubusercontent.com/andrew1407/andrew1407/main/style-guides/common.md`.

| Guide | Applies to | File |
|---|---|---|
| Common (always read first) | every language | `common.md` |
| JavaScript / TypeScript | `.js .jsx .ts .tsx`, `package.json` projects (Node, NestJS, Express, React, Next.js) | `javascript-typescript.md` |
| C++ (general) | `.cpp .hpp .h .cu` outside Unreal projects | `cpp.md` |
| Unreal Engine 5 C++ | `.uproject` / `.uplugin` present, `Source/<Module>/Public` and `Source/<Module>/Private` | `unreal-cpp.md` |
| Unity C# | `.cs` under `Assets/`, `ProjectSettings/`, Zenject | `unity-csharp.md` |
| Go | `.go`, `go.mod` | `go.md` |
| Python | `.py`, `requirements.txt` / `pyproject.toml` | `python.md` |
| Kotlin | `.kt`, `build.gradle(.kts)`, Android | `kotlin.md` |

C++ and Unreal C++ are separate guides. Never mix their conventions: if the project has a
`.uproject` or `.uplugin`, use `unreal-cpp.md`; otherwise use `cpp.md`.

## Workflow

1. **Detect languages.** Scan the project for the file extensions and marker files in the
   table above. Build the list of applicable guides. If none apply, say so and stop.
2. **Read the guides.** Always read `common.md` first, then each applicable language guide in
   full. Fetch every guide from the raw GitHub root. If a fetch fails, retry once, then report
   which guide could not be loaded and continue only with the guides you actually have.
3. **Scope the work.** If the user named files, directories, or a diff, restrict to that.
   Otherwise treat the whole project as in scope, but process it module by module and report
   progress.
4. **Audit before editing.** For each file, list the concrete deviations from the guides:
   naming, file/folder layout, indentation and formatting, architecture (interface/strategy/
   factory, dispatch tables, decomposition), idioms the guide prescribes or forbids.
5. **Refactor and restructure.** Apply the guides in this priority order:
   - Architecture and decomposition rules from `common.md` (interface -> strategy -> factory,
     dispatch tables instead of `switch` over behaviors, one implementation per file, etc.).
   - Language-specific structure: folder taxonomy, file naming, module split.
   - Naming: types, functions, variables, files, suffixes/prefixes the guide mandates.
   - Formatting and prettifying: indentation width, braces, spacing, ordering of members,
     imports, explicit type annotations, and anything else the guide calls out.
   When a file must be renamed or moved, update every import, include, reference, and build
   script that points to it.
6. **Preserve behavior.** This is a style refactor. Do not change public behavior, outputs,
   protocols, or data formats. Keep tests passing; run the project's existing test / build /
   lint commands after edits when they exist and report the result.
7. **Report.** Summarize per language: which guide rules were applied, which files were
   renamed/moved, and any guide rule that was intentionally skipped and why (for example a
   framework constraint that makes it impossible).

## Rules of engagement

- The guides are the source of truth. When a guide and the project's current habit disagree,
  the guide wins, unless following it would break the framework the project relies on. Flag
  such cases instead of silently ignoring them.
- Where a guide says a pattern in older repos should **not** be reproduced (for example tabs in
  older C++ repos), follow the guide's "use this for new code" recommendation.
- Do not introduce new dependencies, formatters, or linters unless the guide names them.
- Do not rewrite third-party or generated code (`node_modules/`, `lib/`, `Intermediate/`,
  `Library/`, vendored sources, generated bindings).
- For languages with no guide in the table, leave code as is and mention it in the report.
- Keep edits minimal per rule: one rule at a time, verified, rather than a wholesale rewrite.

## Example invocations

- "Apply my code style to `server/`" -> detect JS/TS, fetch `common.md` +
  `javascript-typescript.md`, refactor only `server/`.
- "Restructure this Unreal project to my conventions" -> fetch `common.md` + `unreal-cpp.md`,
  mirror `Public/`/`Private/`, apply component and naming rules.
- "Prettify the Python scripts" -> fetch `common.md` + `python.md`, fix naming, module
  taxonomy, `__private` methods, `get_*()` tuple accessors, formatting.
