# Підсистема жестового керування мобільним роботом

Дипломний програмний проєкт для керування мобільним роботом жестами руки. Система
працює з відеопотоком, виділяє ключові точки руки, розпізнає жест і формує команду для
робота.

**Автор:** Ярмоленко Є. М., ІК-23, ФІОТ КПІ ім. Ігоря Сікорського  
**Керівник:** доц., к.т.н. Солдатова М. О.

## Використано

- Python 3.11+ для основної логіки підсистеми.
- OpenCV для роботи з камерою, відеофайлами та підготовки кадрів.
- MediaPipe Hands для знаходження 21 ключової точки руки.
- NumPy для геометричних обчислень над landmarks.
- scikit-learn і joblib для навчених моделей розпізнавання жестів.
- matplotlib для побудови графіків за результатами benchmark.
- pySerial для передавання команд через UART.
- ROS-сумісний sender для інтеграції з `/cmd_vel`.
- Вбудований `http.server` для локальної web-консолі без окремого backend-фреймворку.
- HTML, CSS і JavaScript для демонстраційного web UI.
- pytest, pytest-cov, ruff, black і mypy для тестування та перевірки коду.
- Docker для контейнерного запуску backend-частини.

## Структура репозиторію

```text
src/                 код підсистеми жестового керування
tests/               unit- та інтеграційні тести
scripts/             службові скрипти для запуску, навчання й оцінювання
models/              joblib-моделі для локального демо
data/benchmarks/     CSV-файли з результатами benchmark
docs/                технічна документація, deployment, UX-матеріали
docs/demo/           статичний frontend для GitHub Pages / Netlify / Vercel
```

Сирі датасети, локальні відео, профілі користувачів, кеші, логи та docx-файли не
зберігаються в Git. Локальна структура даних описана в `data/README.md`.

## Встановлення

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

CLI-режим з камерою:

```powershell
python -m src.main --camera 0 --debug --visualize
```

Локальна web-консоль:

```powershell
python -m src.web_ui `
  --camera 0 `
  --host 127.0.0.1 `
  --port 8000 `
  --static-model models/static_gesture_classifier.joblib `
  --dynamic-model models/dynamic_gesture_classifier.joblib
```

Після запуску відкрити `http://127.0.0.1:8000`.

Режим з камерою браузера:

```powershell
python -m src.web_ui `
  --input-mode browser-camera `
  --host 127.0.0.1 `
  --port 8000 `
  --static-model models/static_gesture_classifier.joblib `
  --dynamic-model models/dynamic_gesture_classifier.joblib `
  --cors-origin "*"
```

Запуск на відеофайлі:

```powershell
python -m src.web_ui --video data/test_scenarios/demo.mp4 --loop-video
```

Калібрування користувача:

```powershell
python scripts/calibrate_user.py --user-id operator_01 --camera 0 --samples 5
python -m src.main --camera 0 --debug --visualize `
  --calibration-profile data/user_profiles/operator_01.json
```

## Перевірка

```powershell
python -m pytest tests/ -v
python -m ruff check src tests
python -m black --check src tests
python -m mypy src
```

## Статичний frontend

```powershell
python scripts/export_demo_frontend.py --output docs/demo
```

Варіанти розгортання описані в `docs/DEPLOYMENT.md`.
