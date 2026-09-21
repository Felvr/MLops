# Источник и атрибуция

**MedQuAD — Medical Question Answering Dataset**, Asma Ben Abacha и Dina Demner-Fushman.

- Исходный репозиторий: https://github.com/abachaa/MedQuAD
- Использованный commit: `577bd37b96c02d1833b2c9eed2de9f96964e96cb`.
- Раздел: `3_GHR_QA`, Genetics Home Reference, NLM/NIH.
- Снимок: https://codeload.github.com/abachaa/MedQuAD/zip/577bd37b96c02d1833b2c9eed2de9f96964e96cb
- SHA-256 ZIP: `161d948f8ce8f82accca1f5769a512dff2a01af2b3004d48f2b3da2758a38a09`.
- Лицензия набора: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- Лицензия источника: https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/LICENSE.txt
- Условия CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Авторы просят ссылаться на статью:

> Asma Ben Abacha and Dina Demner-Fushman. A Question-Entailment Approach to
> Question Answering. BMC Bioinformatics 20, 511 (2019).
> https://doi.org/10.1186/s12859-019-3119-4

Исходный README прямо указывает CC BY 4.0. В трёх других разделах MedQuAD
авторы убрали ответы из-за ограничений MedlinePlus; эти разделы не используются.
Текст лицензии сохранён в скачанном архиве; копии источника в Git нет.

Изменения для ДЗ: отбор GHR и четырёх типов вопросов, два вложенных среза
документов, chat-формат, варианты системной инструкции, нормализация пробелов,
фильтрация длин, маскирование распознаваемых ПДн, дедупликация и групповой сплит.
При распространении производного датасета сохраняйте эту атрибуцию и описание
изменений. Поля `source_url`, `xml`, `qid`, `source_revision` в DVC-артефакте
`data/provenance.jsonl` связывают каждую исходную chat-строку с публикацией.
