# Стратегія використання відкритих датасетів

## Мета

Через обмежений час ми не збираємо повний власний датасет для всіх 10 статичних і 3 динамічних
жестів. Натомість використовуємо комбіновану доказову базу:

1. відкриті датасети для кількісної оцінки розпізнавання;
2. малий власний контрольний набір для демонстрації саме нашого сценарію керування роботом;
3. окремий runtime-бенчмарк на локальній камері для FPS, latency та стабільності pipeline.

Такий підхід треба описувати чесно: відкриті набори не є повним еквівалентом середовища
мобільного робота, але вони дають відтворювану базу для F1-score, confusion matrix і порівняння
роботи класифікаторів у різних умовах.

## Основні датасети

| Датасет | Тип даних | Для чого використовуємо | Ліцензія / доступ | Джерело |
|---|---:|---|---|---|
| HaGRID / HaGRIDv2 | RGB-зображення, bbox, hand landmarks | Основна оцінка 10 статичних жестів | CC BY-SA 4.0 variant, великі архіви | \cite{kapitanov2024hagrid}, \cite{nuzhdin2024hagridv2} |
| IPN Hand | RGB-відео 640x480, 30 FPS, continuous annotations | Динамічні жести та false-positive аналіз на continuous stream | CC BY 4.0 | \cite{benitez2021ipnhand} |
| Jester | короткі RGB-кліпи жестів | Додаткова оцінка WAVE_LR, PULL_TOWARD і схожих рухів | Research-use license | \cite{materzynska2019jester} |
| NVGesture | RGB/depth/IR відео динамічних жестів | Опційно для CIRCLE / push-pull, якщо підтверджено доступ і ліцензію | перевірити перед використанням | \cite{molchanov2016online} |
| HANDS | RGB-D статичні HRI-жести | Малий HRI-oriented sanity check для статичних жестів | CC BY 4.0 | \cite{nuzzi2021hands} |

## Мапінг класів

| Наш GestureID | Основне джерело | Кандидати класів у датасетах | Коментар |
|---|---|---|---|
| OPEN_PALM | HaGRIDv2 | `stop`, `palm` | Після візуальної перевірки можна залишити один клас або об'єднати два. |
| FIST | HaGRIDv2 | `fist` | Прямий збіг. |
| THUMB_UP | HaGRIDv2 / Jester | `like`, `Thumb Up` | Для статичного тесту краще HaGRIDv2. |
| THUMB_DOWN | HaGRIDv2 / Jester | `dislike`, `Thumb Down` | Для критичної команди важливо окремо рахувати false positives. |
| INDEX_LEFT | HaGRIDv2 + власний контроль | `point`, `one` + напрямок landmark vector | У HaGRID напрямок не є окремим label, тому left/right треба виводити з landmarks. |
| INDEX_RIGHT | HaGRIDv2 + власний контроль | `point`, `one` + напрямок landmark vector | Потрібна перевірка балансу напрямків. |
| PEACE | HaGRIDv2 | `peace`, `two_up` | Вибрати клас після ручного перегляду прикладів. |
| THREE_FINGERS | HaGRIDv2 | `three`, `three2`, `three3` | Важливо відсіяти варіанти з піднятим великим пальцем. |
| PINKY | HaGRIDv2 | `little_finger` | Прямий збіг у HaGRIDv2. |
| OK_SIGN | HaGRIDv2 | `ok` | Прямий збіг. |
| WAVE_LR | Jester / IPN Hand | `Shaking Hand`, `Swiping Left`, `Swiping Right`, IPN `Throw left/right` | Не всі класи є точним "помахом", тому результати треба називати partial benchmark. |
| CIRCLE | NVGesture / власний контроль | rotating two fingers clockwise/counter-clockwise | Якщо NVGesture недоступний, потрібен малий власний контрольний набір. |
| PULL_TOWARD | Jester / IPN Hand / NVGesture | `Pulling Hand In`, `Zooming In With Full Hand`, IPN `Zoom in` | Найближчий відкритий збіг для руху до камери. |

## Методика використання

1. Завантажити лише потрібні класи, якщо датасет це дозволяє. Для HaGRIDv2 достатньо архівів
   жестів, що відповідають нашому словнику.
2. Не комітити raw data. Розміщувати їх локально в `data/external/`, а похідні landmark CSV/JSON -
   в `data/processed/`.
3. Для статичних зображень використовувати наявні landmarks HaGRIDv2 або проганяти MediaPipe Hands,
   якщо потрібна однакова схема preprocessing.
4. Для відео витягувати 30-кадрові вікна, будувати `TrajectoryBuffer`, а відсутність руки рахувати
   як `UNKNOWN`, а не викидати з оцінки.
5. Розділити calibration/tuning та final evaluation. Якщо є `user_id` або `subject_id`, split робити
   за користувачами, а не випадково по кадрах.
6. У звітах тримати окремі таблиці для:
   - статичних жестів;
   - динамічних жестів;
   - latency/FPS;
   - false positives для `EMERGENCY_STOP`.

## Manifest для оцінювання

Для відтворюваного запуску benchmark використовується CSV-manifest. Один рядок відповідає
одному зразку: зображенню, відео або JSON-файлу з landmarks.

Обов'язкові колонки:

- `path` - шлях до файлу відносно manifest або абсолютний шлях;
- `expected_gesture` - очікуваний жест у форматі `GestureID`, наприклад `OPEN_PALM`.

Додаткові колонки:

- `sample_id` - стабільний ідентифікатор зразка;
- `media_type` - `image`, `video` або `landmarks`; якщо порожньо, тип визначається за розширенням;
- `dataset` - джерело даних, наприклад `hagrid_v2`, `jester`, `own_control`;
- `condition` - умова зйомки або категорія, наприклад `normal`, `low_light`;
- `distance` - дистанція до камери, якщо відома.

Приклад:

```csv
sample_id,path,expected_gesture,media_type,dataset,condition,distance
hagrid_stop_001,../external/hagrid_v2/stop/001.jpg,OPEN_PALM,image,hagrid_v2,normal,unknown
jester_pull_001,../external/jester/pulling_hand_in/001.mp4,PULL_TOWARD,video,jester,normal,unknown
circle_control_001,../processed/landmarks/circle_001.json,CIRCLE,landmarks,own_control,normal,1m
```

Після підготовки manifest запускається:

```powershell
python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/manifest.csv `
  --output data/processed/benchmark_inputs/predictions.csv `
  --classifier-mode auto `
  --frame-stride 2 `
  --max-frames 90 `
  --continue-on-error
```

Результат `predictions.csv` передається у `scripts/benchmark.py`, а згенеровані CSV-файли -
у `scripts/plot_results.py`.

Для навчання моделей на об'єднаному наборі використовується `scripts/retrain_open_data.py`.
Він збирає локальні subset-и відкритих датасетів в один manifest і передає його в
`scripts/train_gesture_models.py`:

```powershell
python scripts/retrain_open_data.py `
  --hagrid-dir data/external/hagrid_v2 `
  --ipn-root data/external/ipn_hand `
  --own-dir data/external/own_control `
  --output-manifest data/processed/training/open_data_manifest.csv `
  --limit-per-class 250 `
  --include-unknown
```

## Обмеження

Відкриті датасети мають інший домен, ніж камера мобільного робота: фон, відстань, роздільна
здатність, положення камери та інструкції до жестів можуть відрізнятися. Тому в ПЗ варто писати,
що ці набори використовуються для відтворюваної оцінки алгоритму, а не як доказ повної готовності
системи до промислової експлуатації. Для демонстрації практичної працездатності залишається
потрібним коротке власне демо-відео: камера -> жест -> команда -> лог/робот.
