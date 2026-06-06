# Unreal Engine 5 C++ Style

Repos: [ReignForce](https://github.com/Andrew1407/ReignForce) (TPS, Enemy AI, EQS/BT, Perception, Enhanced Input,
NNE/ONNX, skills, client-server, tests), [RqstClient](https://github.com/Andrew1407/RqstClient) (Enhanced Input,
HTTP/WS/UDP/TCP), [CartAndPole](https://github.com/Andrew1407/CartAndPole) (NNE, RL),
[ClientConnectionStrategies](https://github.com/Andrew1407/ClientConnectionStrategies) (connection-strategy plugin).
General (non-Unreal) C++ is separate: [cpp.md](./cpp.md).

Read [common.md](./common.md) first.

---

## Source taxonomy

- Strict single-module `Source/<Module>/Public/` + `Private/`, **perfectly mirrored** (every `.h` in
  Public, its `.cpp` at the same sub-path in Private).
- Sub-folders are **feature domains, not technical layers**: `Behavior/{Decorators,EQS,Services,Tasks}`,
  `Enemy/`, `ShooterCharacter/{Animation,Components,Stats}`, `Skills/<category>/`, `Weapons/`,
  `ReinforcementLearning/{Connection,Interfaces,NeuralNetrwork}` (typo preserved), `UI/<screen>/`.
  Pattern: feature → sub-feature → role; `Interfaces/` isolated; `Tests/` top-level.

## Class decomposition philosophy

- **Heavy ActorComponent decomposition**: characters own many components, created with
  `CreateDefaultSubobject<T>(GET_MEMBER_NAME_CHECKED(Owner, Member))` (never a raw string).
  Complex logic components are named `*System` (`UWeaponSlotsSystem`, `UCombatSystem`,
  `UShooterSkillsSystem`); utility ones keep `*Component`.
- **Strategies/models are `UObject` (not Actor)** implementing interfaces: `UHttpClient : UObject, IClient`,
  `UWsClient : UObject, IClient, IConnection`, `UCpuNneModel/UGpuNneModel : UObject, INneModel`.
- **GameState is the orchestrator** (owns spawners, progression, upgrade-state components); GameMode is thin.
- Skills are a `UObject` hierarchy (`UShooterSkillUpgrade`), one `.h/.cpp` per skill, one folder per category.
- Factory as `UObject` (`UClientFactory`) with typed `CreateXxx` + templated private `CreateClient<T>`.

## Naming & suffixes (beyond UE's A/U/F/I/E)

`*System`, `*Component`, `*Runner`, `*Factory`, `*Strategy`, `*Utils` (BlueprintFunctionLibrary),
`*DataAsset`/`*Collection`, `*Upgrade`, `*Progression`, `*HUD`; BT nodes `BTTask_*`/`BTService_*`/`BTDecorator_*`.
- Structs: `F`-prefixed, tight names (`FShooterHealth`, `FClientOptions`, `FRequestData`).
- Enums: `UENUM(BlueprintType) enum class E... : uint8`, `SCREAMING_SNAKE_CASE` values with
  `UMETA(DisplayName=...)`; standalone header, minimal includes.
- Interfaces in `Interfaces/`; UFUNCTION `Category` equals the `I`-name string (`Category="IClient"`).
- Delegates: `FOn<Subject><Event>` (`FOnHealthChanged`, `FOnRoundStarted`); multicast +
  `BlueprintAssignable` for events, single-cast for callbacks.

## UPROPERTY habits

- Private-but-BP-visible via `meta=(AllowPrivateAccess=true)` — the universal pattern.
- Specifier vocabulary: configurable private = `EditAnywhere, BlueprintReadWrite, ...AllowPrivateAccess`;
  created subobject = `VisibleAnywhere, BlueprintReadOnly, ...`; widget = `meta=(BindWidget)`;
  events = `BlueprintAssignable`.
- Numeric clamps on every numeric field: `meta=(ClampMin=0, UIMin=0[, ClampMax=1, UIMax=1, Units="Seconds"])`.
- `EditCondition`/`EditConditionHides` for mutually-exclusive struct fields.
- **Pipe-separated Category hierarchy** matching the class name: `"ShooterHUD|Menu|Skills"`; private
  components all use `"Systems"`.

## UFUNCTION habits

- Getters `BlueprintPure`; stateful `BlueprintCallable`; **interface methods always `BlueprintNativeEvent`**
  (avoids `BlueprintImplementableEvent`); timer/delegate callbacks bare `UFUNCTION()`.
- `FORCEINLINE` getters in the header for component accessors rather than `.cpp` definitions.

## Distinctive intra-class organization

- **`#pragma region UPPER_SNAKE_CASE` ... `#pragma endregion`** everywhere to group members
  (`COMPONENTS`, `EVENTS`, `BLACKBOARD_KEYS`, `SENSE_CONFIGS`, `STATE_CONTROL`, `FACTORY_METHODS`...).
  Combined with the private-component + `GET_MEMBER_NAME_CHECKED` idiom:
  ```cpp
  #pragma region COMPONENTS
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Systems", meta = (AllowPrivateAccess = true))
  TObjectPtr<UWeaponSlotsSystem> WeaponSlotsSystem;
  #pragma endregion

  // in the constructor:
  WeaponSlotsSystem = CreateDefaultSubobject<UWeaponSlotsSystem>(
      GET_MEMBER_NAME_CHECKED(AShooterCharacter, WeaponSlotsSystem));
  ```
- **Anonymous namespace at top of `.cpp`** for module constants (`constexpr`, `constexpr const TCHAR*`);
  named namespaces for utilities (`namespace NneModelUtils`, `namespace TestUtils`).

## Interfaces (decoupling mechanism)

- `UINTERFACE(MinimalAPI)` + `I...` body; all methods `BlueprintNativeEvent`; default boilerplate
  comment kept verbatim; call via `IFoo::Execute_Method(obj, ...)` guarded by a `DoesImplementInterface` check:
  ```cpp
  if (UKismetSystemLibrary::DoesImplementInterface(Obj, UClient::StaticClass()))
      IClient::Execute_Send(Obj, RequestData);
  ```

## AI / Behaviour Tree

- `AShooterAIController` is the hub: creates all `UAISenseConfig_*` subobjects itself; routes perception
  through a `TMap<TSubclassOf<UAISense>, FSenseHandle>` dispatch table + single `OnPerceptionInfoUpdated`.
- Blackboard key names are private `FName` UPROPERTYs (under `#pragma region BLACKBOARD_KEYS`); BT nodes
  take `FBlackboardKeySelector` props instead of hard-coded names.
- Reusable check logic consolidated in a config **struct** (`FShooterAICommandCheck::Check(...)`) reused
  identically across tasks/services/decorators.

## NNE / ONNX integration

- Two-level: interface `INneModel` (or `UNeuralNetworkRunner`) hides CPU vs GPU; a component owns the model
  UObject and dispatches via `Execute_Predict`.
- Newer (ReignForce): `UNeuralNetworkRunner : UObject` holds `TUniquePtr<IModelInstanceCPU>`; queue subclass
  adds `TQueue` + `bModelRunning`; model is `TSoftObjectPtr<UNNEModelData>` async-loaded; inference on
  `AnyNormalThreadNormalTask`, result marshaled back via nested `AsyncTask(ENamedThreads::GameThread, ...)`.

## Networking abstraction

- One `IClient` (+ optional `IConnection`), one enum `EClientLabels`, one config struct `FClientOptions`
  (host/port/delegate/timeout) stored privately in each client. `UClientFactory` creates by label with
  `TSubclassOf<>` overrides. Responses via `DECLARE_DYNAMIC_DELEGATE_TwoParams(FResponseDelegate, ...)`
  stored inside the options struct.
- Test stubs: parallel `U<Protocol>TestClient : U<Protocol>Client, ClientTestData` in `Strategies/Test/`.

## Conventions

- Constructors take `const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get()`.
- Smart pointers: components `TObjectPtr<T>`, non-owned refs `TWeakObjectPtr<T>`, non-UObject resources
  `TSharedPtr`/`TUniquePtr`. `IsValid(x)` before non-trivial use; concise single-line guards.
- Lambdas for HTTP/WS callbacks, timers, async tasks, asset streaming; `MoveTemp` captured payloads.
- JSON: `FJsonObjectConverter::UStructToJsonObjectString` / `JsonObjectStringToUStruct<T>` (clients only).
- SaveGame is a plain data bag; slot management lives in GameState/Player.
- `UPrimaryDataAsset` config bags with `FORCEINLINE BlueprintPure` formatter helpers.

## Tests

- Wrapped in `#if (WITH_AUTOMATION_TESTS && WITH_EDITOR)`. `DEFINE_SPEC` for pure logic;
  `BEGIN_DEFINE_SPEC`/`END_DEFINE_SPEC` with inline helper methods + custom latent commands
  (`FDelayedAction`, `FWaitUntilAction`) for world integration. `TestUtils` namespace for world setup;
  constants in an anonymous namespace; `CastChecked<T>` in tests only.
