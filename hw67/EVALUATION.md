# Оценка ответов и выпуск

Реализация находится в `evaluation/solution.py`, оркестрация — `evaluation/cli.py`.
Полный закреплённый протокол: [docs/protocol.md](docs/protocol.md).

`score(rows, outputs)` сопоставляет ответы по уникальному id и возвращает
общие counts, срезы с counts и per-item. Валиден только JSON с единственным
строковым полем answer из choices; лишние и повторные ключи запрещены.
Отсутствующие/дублирующиеся id — ошибка контракта. Невалидный ответ, timeout
и пустой ответ не исключаются из знаменателя.

`quality_gate(baseline, candidate, critical_slices)` проверяет совместимость
protocol/golden/baseline hashes, status, конкретную модель, id и группы,
конечность метрик и их соответствие counts, отсутствие ошибок генерации,
неухудшение общей accuracy/format rate и accuracy критических срезов,
успех всех обязательных behavioral-кейсов. Любая проблема вызывает ValueError.

Перед выпуском обвязка заново вычисляет score по raw и сверяет сохранённые
метрики и SHA256. Только после PASS назначается champion. Candidate остаётся
отдельным alias и доступен после отказа. Baseline меняется отдельной явной
командой только после принятия и с сохранением истории.

`make check-eval` проверяет синтетические контракты без обучения/сервера.
`make evaluate` генерирует реальные ответы базы и одного кандидата.
`make eval` проверяет уже сохранённые ответы, ненулевой exit при отказе ожидаем.
`make check-negative` портит копию ответов и подтверждает отказ без изменения
весов и alias. Реальные измерения лежат в metrics/ и docs/conclusion.md.

Данные доступны без приватного remote: `make restore-eval`, затем
`uv run dvc status data/eval.dvc`. В docs/evaluation-data.zip находятся golden,
behavioral, полный train-index и audit с оригинальными источниками и разметкой.
