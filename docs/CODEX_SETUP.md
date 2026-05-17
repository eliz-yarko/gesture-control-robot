# 🚀 ІНСТРУКЦІЯ: Як працювати з Codex над дипломом

> Цей файл — твій повний гайд від нуля до робочого workflow з Codex для розробки диплому.

---

## 📌 ЩО ТАКЕ CODEX І ДЛЯ ЧОГО ВІН ТУТ

OpenAI Codex — це AI-агент для написання коду в терміналі або в VS Code. Він може:
- Читати твої файли проєкту як контекст
- Писати, редагувати, видаляти код
- Запускати тести
- Робити коміти в git
- Створювати документацію в Markdown

У твоєму випадку — Codex буде твоїм "молодшим розробником", який пише код і документацію за твоїми інструкціями.

---

## 🎯 ЗАГАЛЬНИЙ WORKFLOW

```
1. Створюєш папку проєкту з AGENTS.md ← головний контекстний файл
2. Запускаєш Codex у цій папці
3. Codex автоматично читає AGENTS.md → розуміє контекст диплому
4. Даєш йому промпт-завдання → він пише код + комітить + оновлює доки
5. Перевіряєш → пушиш у git
```

---

## 📂 КРОК 1. СТВОРИ СТРУКТУРУ ПРОЄКТУ

Відкрий термінал і виконай:

```bash
# Створи папку проєкту
mkdir gesture-control-robot
cd gesture-control-robot

# Ініціалізуй git
git init
git branch -M main

# Створи базову структуру папок
mkdir -p src/{capture,recognition,interpretation,transmission,utils}
mkdir -p tests/fixtures/{hand_landmarks,video_scenarios}
mkdir -p docs/{thesis/figures,diagrams,presentation}
mkdir -p data/{test_scenarios,benchmarks}
mkdir -p scripts
mkdir -p models

# Створи порожні __init__.py для Python-пакетів
touch src/__init__.py
touch src/capture/__init__.py
touch src/recognition/__init__.py
touch src/interpretation/__init__.py
touch src/transmission/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

**Поклади в корінь проєкту файл `AGENTS.md`** (той, що я тобі підготувала окремим файлом).

---

## 🐍 КРОК 2. НАЛАШТУЙ PYTHON-ОТОЧЕННЯ

```bash
# Створи віртуальне середовище
python3 -m venv .venv

# Активуй його
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Створи requirements.txt
cat > requirements.txt << 'EOF'
opencv-python>=4.9.0
mediapipe>=0.10.14
numpy>=1.26.0
scikit-learn>=1.4.0
pyserial>=3.5
matplotlib>=3.8.0
pytest>=8.0.0
pytest-cov>=5.0.0
ruff>=0.4.0
black>=24.0.0
mypy>=1.10.0
EOF

# Встанови залежності
pip install -r requirements.txt
```

---

## 📝 КРОК 3. СТВОРИ .gitignore і README

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Data (великі файли)
data/test_scenarios/*.mp4
data/test_scenarios/*.avi
models/*.h5
models/*.pkl

# Output
*.log
docs/thesis/figures/*.png
!docs/thesis/figures/.gitkeep
EOF

cat > README.md << 'EOF'
# Gesture Control Subsystem for Mobile Robot

Підсистема жестового керування мобільним роботом засобами комп'ютерного зору.

**Автор:** Ярмоленко Є.М., ІК-23, ФІОТ КПІ ім. Ігоря Сікорського
**Керівник:** доц., к.т.н. Солдатова М.О.

## Швидкий старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --camera 0 --debug
```

Деталі — в AGENTS.md
EOF
```

---

## 🤖 КРОК 4. ВСТАНОВИ CODEX

Codex доступний кількома способами. Найзручніше — через VS Code extension або CLI.

### Варіант А: VS Code Extension (рекомендую)

1. Відкрий VS Code
2. Розширення (Ctrl+Shift+X) → шукай "Codex"
3. Встанови офіційне розширення від OpenAI
4. Залогінься через свій ChatGPT акаунт (де є підписка)

### Варіант Б: Codex CLI

```bash
# Через npm
npm install -g @openai/codex

# Авторизація
codex auth login
```

> **Примітка:** конкретні команди для встановлення Codex можуть змінюватись. Перевір актуальну інструкцію на офіційному сайті OpenAI Codex перед встановленням.

---

## 🔥 КРОК 5. ПЕРШИЙ ЗАПУСК CODEX

Відкрий VS Code в папці проєкту:
```bash
code .
```

У VS Code відкрий Codex (зазвичай через значок у бічній панелі або Ctrl+Shift+P → "Codex: New Chat").

**ПЕРШЕ ПОВІДОМЛЕННЯ CODEX (важливо!):**

```
Привіт! Я працюю над дипломним проєктом. У корені проєкту лежить файл AGENTS.md
з повним описом задачі, архітектури, словника жестів, технологічного стеку
та конвенцій коду. Прочитай його перед тим, як ми почнемо.

Підтверди, що ти зрозумів контекст, і запропонуй з якого модуля почати реалізацію.
```

Codex прочитає AGENTS.md і запропонує план. Далі ти підтверджуєш або коригуєш.

---

## 🎬 ПОЕТАПНИЙ ПЛАН РОЗРОБКИ З CODEX

### Етап 1. Базова інфраструктура (1–2 дні)

**Промпт для Codex:**

```
Реалізуй модуль src/config.py з усією конфігурацією проєкту як dataclass.
Включи:
- CameraConfig (індекс камери, розмір кадру, FPS)
- GestureConfig (пороги впевненості, розмір буфера debouncing)
- TransmissionConfig (порт UART, baudrate, або ROS topic)
- LoggingConfig (рівень, формат)

Усі магічні числа з AGENTS.md мають бути тут. Створи unit-тести в tests/test_config.py
для перевірки дефолтних значень. Покажи фінальний код.
```

### Етап 2. Модуль захоплення відео (1 день)

```
Реалізуй src/capture/video_capture.py — клас VideoCapture з методами:
- __init__(config: CameraConfig)
- read() -> Optional[np.ndarray] — повертає попередньо оброблений кадр (resize, mirror, BGR→RGB)
- release() — звільнення камери

Додай unit-тести з мок-камерою. Покрий випадки: камера недоступна, кінець відеофайлу.
```

### Етап 3. Модуль детектування руки (1 день)

```
Реалізуй src/recognition/hand_detector.py — клас HandDetector, що:
- Обгортає MediaPipe Hands
- Метод detect(frame: np.ndarray) -> Optional[HandLandmarks]
- HandLandmarks — dataclass з landmarks (21,3), handedness, confidence

Додай тести з фікстурами зображень рук у різних позах.
```

### Етап 4. Класифікатор статичних жестів (2–3 дні)

```
Реалізуй src/recognition/static_classifier.py — клас StaticGestureClassifier з методом:
- classify(landmarks: HandLandmarks) -> Tuple[GestureID, float]

Реалізуй правила для ВСІХ 10 статичних жестів зі словника в AGENTS.md.
Використовуй обчислення станів пальців (зігнутий/розправлений) та орієнтації.
Винеси геометричні обчислення в src/utils/geometry.py.

Створи фікстури landmarks для кожного з 10 жестів у tests/fixtures/hand_landmarks/.
Напиши тести, що покривають усі 10 жестів + edge cases.
```

### Етап 5. Класифікатор динамічних жестів (2–3 дні)

```
Реалізуй src/recognition/dynamic_classifier.py — клас DynamicGestureClassifier з:
- Буфером траєкторії (deque, max=30 кадрів)
- State machine для 3 динамічних жестів: WAVE_LR, CIRCLE, PULL_TOWARD
- Методом update(landmarks: HandLandmarks) -> Optional[GestureID]

Алгоритми:
- WAVE_LR: детектування зміни напрямку X-координати центру долоні ≥2 разів
- CIRCLE: апроксимація траєкторії кінчика вказівного пальця як кола (через найменші квадрати)
- PULL_TOWARD: монотонна зміна розміру руки (відстані між landmark 0 та 9) на ≥30% за 20 кадрів

Тести з фіксованими послідовностями landmarks.
```

### Етап 6. Command Mapper + Debouncing (1 день)

```
Реалізуй src/interpretation/command_mapper.py — клас CommandMapper з:
- Маппінгом GestureID → RobotCommand (з AGENTS.md)
- Debouncing: команда формується тільки після N=5 послідовних однакових static gesture
  або N=30 для dynamic
- State machine: блокувати взаємно несумісні команди
  (наприклад, INCREASE_SPEED ігнорується якщо робот зупинений)

Тести на debouncing та state-machine логіку.
```

### Етап 7. Передача команд (1 день)

```
Реалізуй три варіанти transmission:
- src/transmission/base_sender.py — абстрактний інтерфейс CommandSender
- src/transmission/serial_sender.py — UART через pySerial
- src/transmission/mock_sender.py — для тестування (просто логує)
- src/transmission/ros_sender.py — ROS (опціонально, через rospy)

Тести на serial з мок-портом, тести на mock_sender.
```

### Етап 8. Головний цикл (1 день)

```
Реалізуй src/main.py — точку входу з CLI (argparse):
- --camera <index> — індекс камери
- --video <path> — або шлях до відеофайлу
- --debug — режим відлагодження з overlay
- --transmission {serial,ros,mock} — вибір транспорту
- --port <path> — порт для serial

Зв'яжи усі модулі в pipeline. Додай вимірювання FPS та latency через src/utils/metrics.py.
```

### Етап 9. Бенчмарк і графіки (1–2 дні)

```
Реалізуй scripts/benchmark.py:
- Прогоняє ВСІ відео з data/test_scenarios/ через pipeline
- Для кожного жесту вимірює: Precision, Recall, F1, latency
- Зберігає результати у data/benchmarks/results.csv

Реалізуй scripts/plot_results.py:
- Читає CSV → генерує:
  - Confusion matrix (heatmap)
  - Bar chart F1 за умовами освітлення
  - Histogram latency
  - FPS over time
- Зберігає PNG у docs/thesis/figures/
```

### Етап 10. Документація розділів ПЗ (паралельно)

Після кожного етапу:
```
На основі реалізованого модуля [назва] напиши/онови розділ [3.X] у docs/thesis/03_design.md.
Стиль — академічний український, з посиланнями на джерела з docs/references.bib.
Включи:
- Опис призначення модуля
- Алгоритм / формули
- Діаграму (PlantUML у docs/diagrams/)
- Приклад коду (4–10 рядків з реалізації)
```

---

## 💬 ПРОМПТ-ШАБЛОНИ ДЛЯ CODEX

Збережи їх — будуть рятувати час.

### 🔧 Реалізація фічі

```
[Опиши, що треба зробити]

Слідуй конвенціям з AGENTS.md:
- type hints + Google docstrings
- unit-тести в tests/
- conventional commit після завершення

Покажи план перед тим як писати код.
```

### 🐛 Виправлення бага

```
Я знайшла баг: [опис]. Очікувана поведінка: [...], фактична: [...].
Знайди причину, виправ і додай регресійний тест. Не змінюй інші тести,
щоб баг "не проявлявся".
```

### 📝 Документування

```
Напиши розділ [N.X] у docs/thesis/[file].md на тему [тема].
Обсяг: 1.5–2 сторінки. Використовуй академічний український стиль.
Цитуй джерела з docs/references.bib (формат: \cite{key}).
Якщо потрібна діаграма — згенеруй PlantUML у docs/diagrams/[name].puml
та згадай її як "(див. рис. X.Y)".
```

### 📊 Аналіз результатів

```
Запусти scripts/benchmark.py на свіжих даних. Згенеруй графіки.
На основі результатів сформуй короткий висновок (3–5 речень):
- Чи досягнуті цільові метрики з AGENTS.md (NFR)?
- Які жести розпізнаються найгірше і чому?
- Які умови (освітлення/відстань) дають найбільшу деградацію?

Запропонуй 2–3 покращення.
```

### 🔄 Git-операції

```
Зроби коміт усіх змін з повідомленням у форматі Conventional Commits.
Якщо є логічно різні зміни — зроби кілька комітів.
Перевір, що ми НЕ в гілці main; якщо в main — створи нову гілку feature/<name>.
```

### 🧪 Тестування

```
Запусти pytest tests/ -v --cov=src. Якщо є падіння — поясни кожне.
Якщо покриття < 70% — покажи які модулі недостатньо покриті
і запропонуй які тести додати.
```

---

## 🎓 РОБОТА З ДИПЛОМОМ (ПЗ + ДІАГРАМИ + ГРАФІКИ)

### Загальний підхід

1. **Код пишемо паралельно з документацією** — щойно реалізував модуль, одразу проси Codex написати/оновити відповідний підрозділ у `docs/thesis/`
2. **Markdown — основа** — пишеш у MD, потім конвертуєш у docx через pandoc на фінальному етапі
3. **Цитати — тільки з `references.bib`** — Codex не вигадує джерела, бо у AGENTS.md це заборонено
4. **Діаграми — PlantUML** — текстовий формат, версіюється в git, легко рендериться

### Конвертація MD → DOCX (для здачі)

```bash
# Установи pandoc (один раз)
# macOS: brew install pandoc
# Ubuntu: sudo apt install pandoc
# Windows: https://pandoc.org/installing.html

# Конвертація всього диплому в один docx
pandoc docs/thesis/0*.md docs/thesis/99_*.md \
    --bibliography=docs/references.bib \
    --csl=docs/dstu8302.csl \
    --reference-doc=docs/template.docx \
    -o "ІК-23 Ярмоленко диплом.docx"
```

### Шаблон промпту "напиши розділ"

```
Напиши розділ 2 (Аналіз предметної області) у docs/thesis/02_analysis.md.

Структура:
2.1. Жестове керування як спосіб HCI/HRI
2.2. Огляд існуючих рішень (DJI Tello, MediaPipe Hands, OpenPose, HandTrack.js)
2.3. Класифікація методів CV для розпізнавання жестів
2.4. Постановка задачі

Вимоги:
- 4-6 сторінок (≈1500–2000 слів)
- Академічний український стиль
- Кожне твердження з цифрами/фактами — з посиланням на джерело з references.bib
- Включи порівняльну таблицю методів
- Запропонуй 2-3 PlantUML-діаграми, якщо доречно

Перед написанням покажи план.
```

---

## ⚠️ ВАЖЛИВО: БЕЗПЕКА І РИЗИКИ

1. **Codex має доступ до твоєї файлової системи** — він може видаляти/змінювати файли. Працюй у git, щоб все відкочувати при потребі.
2. **Не давай йому пушити в main** — у AGENTS.md це заборонено, але перевіряй сама
3. **Чекай і перевіряй** — особливо команди типу `rm -rf`, `git push --force`
4. **Не довіряй цитатам** — якщо Codex видумав джерело, він порушив правила. Перевір `references.bib` що цитата є там
5. **Не дозволяй "вгадувати" — нехай задає питання** — якщо щось неясно, краще хай уточнить, ніж напише невірно

---

## 📋 ЩОДЕННИЙ ЧЕК-ЛИСТ

Перед початком сесії:
- [ ] Зробити `git pull` (якщо працюєш з кількох комп'ютерів)
- [ ] Активувати venv: `source .venv/bin/activate`
- [ ] Відкрити Codex і нагадати про AGENTS.md

Після сесії:
- [ ] `pytest tests/ -v` — переконатися, що тести проходять
- [ ] `ruff check src/` — лінтинг
- [ ] `git status` — нічого зайвого не закомічено
- [ ] `git push origin <branch>` — зберегти прогрес

---

## 🆘 ЯКЩО ЩОСЬ ПІШЛО НЕ ТАК

| Проблема | Рішення |
|---|---|
| Codex переписав файл, який не треба було | `git checkout -- <file>` — повернути з останнього коміту |
| Codex закомітив щось небажане | `git reset HEAD~1` — скасувати останній коміт (зміни залишаться) |
| Codex запустив команду, що зламала проєкт | `git stash` → `git pull` — повернути до останньої стабільної |
| Codex видумав джерело | Покажи йому AGENTS.md, нагадай заборону. Видали цитату. |
| Тести почали падати після рефакторингу | Не змінюй тести! Поверни код назад через `git diff` |
| Не знаю, з чого починати | "Запропонуй наступний крок згідно з планом в AGENTS.md" |

---

## 📞 ДОПОМОГА

- AGENTS.md — повна інструкція для агента
- references.bib — допустимі джерела для цитування
- README.md — швидкий старт

**Не соромся повертатись до Claude (мене) для:**
- Обговорення архітектури перед кодуванням
- Розбору важких алгоритмів
- Перевірки сумнівних рішень Codex
- Структурування розділів ПЗ
- Підготовки до захисту

Codex — для код-роботи, я — для стратегічних і навчальних питань. Це різні інструменти, і вони доповнюють одне одного. 💪

---

*Версія 1.0 — травень 2026*
