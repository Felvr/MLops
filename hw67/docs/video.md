# Сценарий видео, 5–8 минут

Снимать из папки hw67. Docker Desktop должен работать. Повторно обучать модели
для записи не нужно. Открыть http://127.0.0.1:5006 и docs/conclusion.md.

1. **0:00–0:50.** Показать структуру одного репозитория, params.yaml,
   docs/defects.md. Объяснить четыре причины и исправления.
2. **0:50–2:10.** В MLflow открыть hw6-lora, sweep и шесть дочерних runs.
   Показать lr/r/modules, общий бюджет 4 шага, кривые val до/после,
   хеши train/val, pinned base revision, артефакты adapter/tokenizer/manifest.
   Открыть comparison.csv и объяснить выбор по минимальному val loss.
3. **2:10–3:10.** В реестре показать candidate. Открыть protocol.md и
   data/eval/golden.jsonl: 50 вопросов, source, expected, четыре служебных
   поля; отдельно три среза и behavioral. Показать полный train-index
   и data-provenance.json. expected не включается в prompt.
4. **3:10–4:30.** Показать conclusion.md, baseline/candidate accuracy,
   format rate, counts срезов, diff и четыре пары сырых ответов.
   Объяснить, что правильный текст вне JSON не проходит строгий контракт.
5. **4:30–5:20.** Выполнить `make eval`. Если код возврата ненулевой,
   объяснить причину по quality-gate.json: это корректный отказ в выпуске.
   Champion не назначать после отказа. При PASS показать released.json
   и alias champion. Выполнить `make check-negative`: испорчена копия ответов,
   реальные веса и alias сохранены.
6. **5:20–6:50.** Выполнить `make check-storage`: force-recreate и одинаковый
   SHA256 probe. Затем `make check-all`: 9/9 проверок, fingerprint и тесты
   оценки. Проверка 7 загружает candidate в отдельном процессе вне проекта.
7. **6:50–7:30.** Вывод: короткий учебный поиск, простой golden на чтение,
   ограничения текстового сходства и model judge; качество нельзя приравнивать
   к val loss или клинической безопасности. Видео сохранить отдельно;
   при необходимости добавить docs/check.mp4 перед финальной сдачей.

На защите: run — один запуск; experiment — группа запусков; artifact — файл;
backend — метаданные; volume — постоянные файлы контейнера; registry version —
закреплённый пакет; alias — изменяемый указатель; candidate — лучший по loss;
champion — версия, принятая ещё и по качеству ответов; golden — регрессионный
контракт, который после многократного использования не является независимым test.
