# Synthetic demo data

> Увага: це штучний демонстраційний приклад. Його не можна подавати в дипломі як дані
> реального UX-дослідження. Мета файлу - показати структуру таблиць і те, як можуть
> виглядати агреговані результати після реального збору даних.

## Приклад: вступна таблиця учасників

| participant_id | role | technical_experience | dominant_hand | notes |
|---|---|---|---|---|
| P01 | technical student | medium | right | synthetic |
| P02 | warehouse worker | low | right | synthetic |
| P03 | medical worker | low | right | synthetic |
| P04 | engineer | high | left | synthetic |
| P05 | UX designer | medium | right | synthetic |
| P06 | robotics student | high | right | synthetic |

## Приклад: картковий метод

| command | top_selected_gesture | selection_count | mean_intuitiveness | confusion_risk |
|---|---|---:|---:|---|
| STOP | OPEN_PALM | 6 | 4.8 | low |
| FORWARD | FIST | 4 | 3.9 | medium |
| START | THUMB_UP | 5 | 4.6 | low |
| EMERGENCY_STOP | THUMB_DOWN | 4 | 4.1 | low |
| TURN_LEFT | INDEX_LEFT | 6 | 4.7 | medium |
| TURN_RIGHT | INDEX_RIGHT | 6 | 4.7 | medium |
| INCREASE_SPEED | PEACE | 4 | 3.8 | medium |
| DECREASE_SPEED | THREE_FINGERS | 3 | 3.2 | medium |
| RETURN_HOME | PINKY | 2 | 2.8 | high |
| CONFIRM_ACTION | OK_SIGN | 6 | 4.9 | low |
| MODE_TOGGLE | WAVE_LR | 4 | 3.7 | medium |
| ROTATE_360 | CIRCLE | 5 | 4.4 | low |
| APPROACH_OPERATOR | PULL_TOWARD | 5 | 4.2 | medium |

## Приклад: usability tasks

| mode | task_completion_rate | mean_time_seconds | mean_attempts | false_commands |
|---|---:|---:|---:|---:|
| A_no_feedback | 0.78 | 5.9 | 2.1 | 7 |
| B_with_feedback | 0.91 | 4.2 | 1.4 | 3 |

## Приклад: questionnaire scores

| mode | mean_sus | mean_nasa_tlx | mean_borg_cr10 |
|---|---:|---:|---:|
| A_no_feedback | 68.3 | 9.8 | 3.5 |
| B_with_feedback | 81.7 | 6.4 | 2.8 |

## Як це перетворити на реальні дані

1. Замінити participant_id на реальних анонімних учасників `P01`, `P02`, ...
2. Заповнювати таблиці під час або одразу після тестів.
3. Зберігати сирі дані окремо від агрегованих.
4. У дипломі вказати фактичну кількість учасників і процедуру збору.
5. Не використовувати цей synthetic-файл як джерело результатів.
