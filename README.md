# Gesture Control Subsystem for Mobile Robot

Підсистема жестового керування мобільним роботом засобами комп'ютерного зору.

**Автор:** Ярмоленко Є. М., ІК-23, ФІОТ КПІ ім. Ігоря Сікорського  
**Керівник:** доц., к.т.н. Солдатова М. О.

## Поточний стан

Репозиторій переведено з порожнього scaffold у foundation-рівень дипломної системи:

- визначено доменні типи жестів і команд;
- додано OpenCV-ready модуль захоплення та preprocessing кадрів;
- додано MediaPipe-ready wrapper для детектування 21 ключової точки руки;
- реалізовано rule-based класифікатор 10 статичних жестів;
- реалізовано baseline-класифікатор 3 динамічних жестів через буфер траєкторії;
- додано adaptive calibration і JSON-профілі користувачів;
- додано frame-to-command pipeline, debouncing і map жестів у команди робота;
- підготовлено mock/UART/ROS транспортний шар;
- додано перші unit-тести та архітектурну документацію.

## Швидкий старт

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Перевірка якості

```powershell
python -m pytest tests/ -v --cov=src --cov-report=term-missing
python -m ruff check src tests
python -m black --check src tests
python -m mypy src
```

Запуск прототипу з камерою після встановлення залежностей:

```powershell
python -m src.main --camera 0 --debug --visualize
```

Збір персонального профілю калібрування:

```powershell
python scripts/calibrate_user.py --user-id operator_01 --camera 0 --samples 5
python -m src.main --camera 0 --debug --visualize --calibration-profile data/user_profiles/operator_01.json
```

## Benchmark

Для оцінювання на subset-ах відкритих датасетів використовується manifest CSV:

```csv
sample_id,path,expected_gesture,media_type,dataset,condition,distance
hagrid_001,../external/hagrid_v2/stop/001.jpg,OPEN_PALM,image,hagrid_v2,normal,1m
jester_001,../external/jester/pulling_hand_in/001.mp4,PULL_TOWARD,video,jester,normal,1m
circle_001,../processed/landmarks/circle_001.json,CIRCLE,landmarks,own_control,normal,1m
```

Швидкий варіант для власного контрольного набору:

```powershell
python scripts/record_test_video.py --gesture OPEN_PALM --seconds 3
python scripts/record_test_video.py --gesture PULL_TOWARD --seconds 4

python scripts/build_manifest.py `
  --input data/external/own_control `
  --dataset own_control `
  --output data/processed/benchmark_inputs/manifest.csv `
  --condition normal `
  --distance 1m
```

Побудова `predictions.csv`:

```powershell
python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/manifest.csv `
  --output data/processed/benchmark_inputs/predictions.csv `
  --classifier-mode auto `
  --frame-stride 2 `
  --max-frames 90 `
  --continue-on-error
```

Підрахунок метрик і графіків:

```powershell
python scripts/benchmark.py `
  --input data/processed/benchmark_inputs/predictions.csv `
  --output data/benchmarks/results.csv `
  --group-by dataset condition distance

python scripts/plot_results.py `
  --input data/benchmarks/results.csv `
  --output docs/thesis/figures/
```

## Структура

- `src/` — код підсистеми жестового керування.
- `tests/` — unit- та інтеграційні тести.
- `docs/thesis/` — Markdown-розділи пояснювальної записки.
- `docs/diagrams/` — PlantUML-діаграми.
- `docs/ux_research/` — UX-план, personas, протоколи інтерв'ю та usability test.
- `docs/references.bib` — дозволена бібліографія для цитування.
- `docs/CODEX_SETUP.md` — інструкція workflow для роботи з Codex.

Повний контекст проєкту, правила кодування та словник жестів описані в `AGENTS.md`.
