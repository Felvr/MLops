"""Build the submission report from measured artifacts only."""
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def inline(value):
    return str(value).replace('\n', r'\n').replace('|', r'\|')


def main():
    rows = list(csv.DictReader(Path('docs/comparison.csv').open()))
    best = min(rows, key=lambda r: float(r['final_val_loss']))
    baseline = read('metrics/baseline-eval.json')
    candidate = read('metrics/candidate-eval.json')
    gate = read('metrics/quality-gate.json')
    negative = read('metrics/negative-gate.json')
    diff = read('metrics/eval-diff.json')
    storage = read('metrics/storage-check.json')
    golden = {r['id']:r for r in map(json.loads, Path('data/eval/golden.jsonl').read_text().splitlines())}
    a = {r['id']:r for r in baseline['per_item']}
    b = {r['id']:r for r in candidate['per_item']}
    lines = ['# Итоги ДЗ 6+7', '',
             '## A. Эксперименты и кандидат', '',
             'Измерения получены локально на Intel macOS, CPU float32, Qwen3-0.6B. '
             'Это короткий учебный поиск: 6 конфигураций × 4 шага, train 32, val 8, '
             'batch 1 × accumulation 2: 8 микробатчей на кандидата из пула 32. '
             'Целые примеры ДЗ5 сохранены без перетокенизации; '
             'SHA256 и связь с ДЗ4 — в data-provenance.json. Test в loss-подборе не участвует.', '',
             f"Минимальный val loss: **{best['name']}**, {float(best['base_val_loss']):.6f} → "
             f"{float(best['final_val_loss']):.6f}; ratio = "
             f"{float(best['final_val_loss']) / float(best['base_val_loss']):.6f} при пороге 1.0.", '',
             f"Run: `{best['run_id']}`. Конкретный кандидат: `{candidate['model']['uri']}`.", '',
             'Сравнение всех запусков — comparison.csv. seconds — время обучения и val; '
             'peak_memory_mb — максимум RSS на точках измерения (не непрерывный пик и не VRAM); '
             'adapter_mb включает tokenizer и manifest. Разброс времени зависит от прогрева и нагрузки. '
             'Кандидаты обучались последовательно в одном процессе: RSS зависит также от кешей '
             'аллокатора и не позволяет приписать всю разницу только LoRA. '
             'Один seed и маленький val не доказывают статистического превосходства.', '',
             '## B. Ответы двух версий', '',
             'Протокол и критические срезы закреплены до генерации в protocol.md и evaluation/params.json. '
             'Задача — чтение медицинской выдержки, не клиническая диагностика. '
             '50 документов вне полного исходного train; один и тот же greedy prompt и лимит 16 токенов.', '',
             'Baseline: `' + json.dumps(baseline['model'],ensure_ascii=False) + '`.', '',
             '| Набор | Версия | n | correct | valid | errors | accuracy | format rate |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ['overall'] + sorted(baseline['slices']):
        for role, report in [('baseline',baseline), ('candidate',candidate)]:
            m = report['overall'] if name=='overall' else report['slices'][name]
            lines.append(f"| {name} | {role} | {m['n']} | {m['correct']} | {m['valid']} | {m['errors']} | {m['accuracy']:.4f} | {m['format_rate']:.4f} |")
    lines += ['', 'Исправлены id: ' + (', '.join(diff['improved']) or 'нет') + '.',
              'Потеряны id: ' + (', '.join(diff['regressed']) or 'нет') + '.', '',
              '### Обязательное поведение', '',
              '| Кейс | Эталон | Baseline raw | Candidate raw | Candidate correct |', '|---|---|---|---|---|']
    behavior_a = {r['id']:r for r in baseline['behavioral']['per_item']}
    behavior_expected = {r['id']:r['expected'] for r in map(json.loads, Path(read('evaluation/params.json')['behavioral']).read_text().splitlines())}
    for item in candidate['behavioral']['per_item']:
        raw_a = inline(behavior_a[item['id']]['raw'])
        raw_b = inline(item['raw'])
        lines.append(f"| {item['id']} | `{behavior_expected[item['id']]}` | `{raw_a}` | `{raw_b}` | {item['correct']} |")
    lines += ['', '### Разбор четырёх пар', '']
    source_notes = {
        'med-01': 'В выдержке сказано, что состояние присутствует при рождении. '
                  'Это непосредственно поддерживает A. Варианты про ANTXR2 и GJB2 '
                  'не подтверждены данным текстом; для ответа не требуется угадывать болезнь.',
        'med-19': 'Выдержка содержит частоту примерно 1 на 10 000. Это вариант A; '
                  'два других варианта описывают гены, о которых здесь ничего не сказано. '
                  'Пример показывает лёгкость среза frequency, а не медицинскую экспертизу.',
        'med-35': 'Источник прямо называет ANTXR2 и его прежнее имя CMG2, что соответствует B. '
                  'GJB2 и SHOX относятся к другим выдержкам. Совпадение точного имени гена '
                  'здесь полезно, но не заменяет понимание причинной связи в открытом ответе.',
        'med-50': 'Текст описывает накопление мутаций в генах роста, деления и восстановления ДНК. '
                  'Это поддерживает B, а не конкретные гены ANTXR2/GJB2 из отвлекающих вариантов. '
                  'Успех на этой паре не проверяет распознавание заболевания по симптомам.'
    }
    chosen = list(dict.fromkeys(diff['regressed'][:2] + diff['improved'][:2] + ['med-01','med-19','med-35','med-50']))[:4]
    for key in chosen:
        lines += [f"**{key}**: эталон `{golden[key]['expected']}`, срез {', '.join(golden[key]['slices'])}.", '',
                  'Baseline: `' + inline(a[key]['raw']) + '`.', '',
                  'Candidate: `' + inline(b[key]['raw']) + '`.', '']
        if a[key]['correct'] and b[key]['correct']:
            note='Обе версии выбрали утверждение, прямо взятое из источника. Это успех чтения, но не доказательство медицинского рассуждения.'
        elif not b[key]['valid']:
            note='Кандидат нарушил строгий формат или допустимые метки. Даже полезный текст не заменяет контракт JSON, поэтому результат равен нулю.'
        elif b[key]['correct']:
            note='Кандидат выбрал подтверждённый источником вариант; baseline ошибся. Это локальное исправление, которое нужно оценивать вместе с потерянными ответами.'
        else:
            note='Кандидат выбрал неверный вариант. Совпадение медицинских терминов с выдержкой само по себе не делает утверждение верным.'
        lines += [source_notes.get(key, 'Эталон проверяется по оригинальной выдержке в annotation-audit.jsonl.'), '', note, '']
    passed = gate.get('passed') is True
    failed_behavior = [x['id'] for x in candidate['behavioral']['per_item'] if not x['correct']]
    lines += ['Непройденные behavioral-кейсы кандидата: ' + (', '.join(failed_behavior) or 'нет') + '.', '']
    lines += ['## Решение о выпуске', '',
              ('**PASS**: кандидат удовлетворил заранее заданному правилу.' if passed else '**Отказ**: ' + gate.get('reason','нет подтверждения PASS') + '.'), '',
              'Негативная проверка: rejected=' + str(negative['rejected']) + '; ' + negative['reason'] + '.', '',
              'Probe пережил force-recreate контейнера; SHA256 `' + storage['sha256'] + '`; run `' + storage['run_id'] + '`.', '']
    if Path('metrics/released.json').exists():
        lines += [f"Champion: `models:/{candidate['model']['name']}@champion`; "
                  f"eval run `{candidate['evaluation_run_id']}`. Назначен после PASS; "
                  'подтверждение: metrics/released.json.', '']
    else:
        lines += ['Champion не назначен. Candidate остаётся доступным, baseline не менялся. '
                  'Следующий шаг — отдельный эксперимент по корректному отказу вне темы; '
                  'новый протокол потребует новой оценки обеих версий. Порог по текущим ответам не менялся.', '']
    lines += ['## Ограничения', '',
              'Golden содержит простые source-grounded вопросы: правильный вариант повторяет предложение '
              'выдержки, distractors взяты из других документов. Такой набор легче реальных медицинских '
              'вопросов, может решаться сопоставлением строк и не покрывает безопасность клинического применения. '
              'Разметка проверена по сохранённому источнику, а не актуализирована врачом. '
              'После использования в gate этот golden не является независимым итоговым аудитом.', '',
              'Точное сравнение метки проверяет выбранный факт в пределах заданных вариантов, '
              'но ничего не говорит о качестве свободного объяснения. BLEU/ROUGE вознаграждают совпадения слов '
              'и могут высоко оценить фактически неверное утверждение. BERTScore тоже не является проверкой фактов. '
              'LLM-as-judge полезен для открытых ответов по рубрике, но подвержен эффектам порядка, '
              'многословия и предпочтения знакомого стиля, может ошибаться вместе с оцениваемой моделью. '
              'Для него нужны слепые A/B и B/A и проверка человеком; для текущих меток judge не нужен.', '',
              'Автоматический контроль нулевого пересечения текстов и group_id не исключает смысловых '
              'перефразов или знания MedQuAD базовой моделью. Полный train-index и источники '
              'доступны в docs/evaluation-data.zip; восстановление: make restore-eval.', '']
    check_log = Path('docs/check-all.log')
    if check_log.exists():
        evidence = check_log.read_text()
        if 'Итог: 9/9' in evidence and 'Ran 24 tests' in evidence and 'Ran 5 tests' in evidence and 'FAIL ' not in evidence:
            lines += ['## Итоговая приёмка', '',
                      'Пройдены 9/9 проверок MLflow, 24 преподавательских и 5 дополнительных тестов. '
                      'Отпечаток tests/: c42ac7788026. Журнал: [check-all.log](check-all.log). '
                      'Повторная проверка хранения после изменения workers: [storage-check.log](storage-check.log).', '']
    lines += ['## Runs', '',
              'Локальный UI после make up: http://127.0.0.1:5006. Это локальные ссылки, '
              'для другого компьютера нужно повторить эксперимент; сохранённые отчёты доступны без сервера.', '']
    sweep = read('metrics/sweep.json')
    from mlflow import MlflowClient
    from src.common import connect, load_params
    params = load_params()
    connect(params)
    client = MlflowClient()
    ids = dict(sweep=sweep['parent_id'], evaluation=candidate['evaluation_run_id'],
               **{r['name']:r['run_id'] for r in rows})
    links = {}
    for name, run_id in ids.items():
        experiment = client.get_run(run_id).info.experiment_id
        links[name] = params['tracking']['uri'] + f'/#/experiments/{experiment}/runs/{run_id}'
        lines.append(f'- [{name}]({links[name]}): `{run_id}`')
    for name in ('quality-gate.json', 'negative-gate.json', 'negative-outputs.json', 'storage-check.json'):
        client.log_artifact(candidate['evaluation_run_id'], f'metrics/{name}', 'evaluation')
    Path('docs/conclusion.md').write_text('\n'.join(lines)+'\n')
    Path('docs/run-links.json').write_text(json.dumps(links,indent=2)+'\n')



if __name__ == '__main__':
    main()
