# Gesture Control Subsystem for Mobile Robot

Підсистема жестового керування мобільним роботом засобами комп'ютерного зору.

**Автор:** Ярмоленко Є. М., ІК-23, ФІОТ КПІ ім. Ігоря Сікорського  
**Керівник:** доц., к.т.н. Солдатова М. О.

## Швидкий Старт

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.main --camera 0 --debug
```

## Структура

- `src/` — код підсистеми жестового керування.
- `tests/` — unit- та інтеграційні тести.
- `docs/thesis/` — Markdown-розділи пояснювальної записки.
- `docs/diagrams/` — PlantUML-діаграми.
- `docs/references.bib` — дозволена бібліографія для цитування.
- `docs/CODEX_SETUP.md` — інструкція workflow для роботи з Codex.

Повний контекст проєкту, правила кодування та словник жестів описані в `AGENTS.md`.
