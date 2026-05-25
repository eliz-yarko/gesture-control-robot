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

Запуск прототипу з камерою після встановлення залежностей:

```powershell
python -m src.main --camera 0 --debug --visualize
```

## Структура

- `src/` — код підсистеми жестового керування.
- `tests/` — unit- та інтеграційні тести.
- `docs/thesis/` — Markdown-розділи пояснювальної записки.
- `docs/diagrams/` — PlantUML-діаграми.
- `docs/references.bib` — дозволена бібліографія для цитування.
- `docs/CODEX_SETUP.md` — інструкція workflow для роботи з Codex.

Повний контекст проєкту, правила кодування та словник жестів описані в `AGENTS.md`.
