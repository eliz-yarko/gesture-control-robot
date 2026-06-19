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
| HANDS | RGB-D статичні HRI-жести, bbox-анотації | Зовнішній sanity check для статичних жестів і HRI-сценарію | CC BY 4.0 | \cite{nuzzi2021hands} |
| IPN Hand | RGB-відео 640x480, 30 FPS, continuous annotations | Динамічні жести та false-positive аналіз на continuous stream | CC BY 4.0 | \cite{benitez2021ipnhand} |
| Jester | короткі RGB-кліпи жестів | Додаткова оцінка WAVE_LR, PULL_TOWARD і схожих рухів | Research-use license | \cite{materzynska2019jester} |
| NVGesture | RGB/depth/IR відео динамічних жестів | Опційно для CIRCLE / push-pull, якщо підтверджено доступ і ліцензію | перевірити перед використанням | \cite{molchanov2016online} |

## Мапінг класів

| Наш GestureID | Основне джерело | Кандидати класів у датасетах | Коментар |
|---|---|---|---|
| OPEN_PALM | HANDS / власний контроль | `span`, `open_palm`, `palm`, `five` | У HANDS це найближчі відкриті долоні; фінальний тест лишається на власному контрольному наборі. |
| FIST | HANDS / власний контроль | `zero`, `digit_0`, `fist` | Для HRI-датасету може бути подано як жест цифри 0 або окремий class folder. |
| THUMB_UP | власний контроль / Jester | `thumb_up`, `Thumb Up` | Якщо в зовнішньому статичному наборі немає точного збігу, клас оцінюється на власних зразках. |
| THUMB_DOWN | власний контроль / Jester | `thumb_down`, `Thumb Down` | Для критичної команди важливо окремо рахувати false positives. |
| INDEX_LEFT | HANDS + власний контроль | `point_left`, `pointing_left`, `direction_left` | Напрямок краще брати з явно розмічених directional класів або з landmark vector. |
| INDEX_RIGHT | HANDS + власний контроль | `point_right`, `pointing_right`, `direction_right` | Потрібна перевірка балансу напрямків. |
| PEACE | HANDS / власний контроль | `two`, `digit_2` | Наближений збіг через двопальцевий статичний жест. |
| THREE_FINGERS | HANDS / власний контроль | `three`, `digit_3` | Наближений збіг через трипальцевий статичний жест. |
| PINKY | власний контроль | `pinky`, `little_finger` | Якщо зовнішній dataset не має класу, залишити тільки власний контроль. |
| OK_SIGN | HANDS / власний контроль | `ok`, `ok_sign` | Використовувати лише за наявності точного class folder у локальному subset. |
| WAVE_LR | Jester / IPN Hand | `Shaking Hand`, `Swiping Left`, `Swiping Right`, IPN `Throw left/right` | Не всі класи є точним "помахом", тому результати треба називати partial benchmark. |
| CIRCLE | NVGesture / власний контроль | rotating two fingers clockwise/counter-clockwise | Якщо NVGesture недоступний, потрібен малий власний контрольний набір. |
| PULL_TOWARD | Jester / IPN Hand / NVGesture | `Pulling Hand In`, `Zooming In With Full Hand`, IPN `Zoom in` | Найближчий відкритий збіг для руху до камери. |

## Методика використання

1. Завантажити лише потрібні класи, якщо датасет це дозволяє. Для HANDS достатньо RGB subset-у
   класів, що мають прямий або обґрунтований наближений збіг із нашим словником.
2. Не комітити raw data. Розміщувати їх локально в `data/external/`, а похідні landmark CSV/JSON -
   в `data/processed/`.
3. Для статичних зображень проганяти MediaPipe Hands, якщо потрібна однакова схема preprocessing
   між HANDS, власним контрольним набором та іншими джерелами.
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
- `dataset` - джерело даних, наприклад `hands`, `jester`, `own_control`;
- `condition` - умова зйомки або категорія, наприклад `normal`, `low_light`;
- `distance` - дистанція до камери, якщо відома.

Приклад:

```csv
sample_id,path,expected_gesture,media_type,dataset,condition,distance
hands_span_001,../external/hands/span/001.jpg,OPEN_PALM,image,hands,normal,unknown
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
  --hands-dir data/external/hands `
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
