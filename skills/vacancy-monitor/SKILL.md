---
name: vacancy-monitor
description: "Консольный мониторинг, многофакторный скоринг и фильтрация вакансий (HH, Habr, Telegram, dry-run)."
---

# Руководство по мониторингу и фильтрации вакансий (Career Search Kit)

Инструмент консольного мониторинга, автоматического скоринга и фильтрации вакансий по целевым направлениям:
1. **Системный аналитик / Аналитик интеграций** (REST API, Postman, Swagger, SQL, BPMN, UML, 1C/CRM, ERD)
2. **Специалист по AI-автоматизации / No-code интегратор** (n8n, Make, AI-агенты, LLM, LangChain, webhooks, Python)

Инструмент поддерживает как сетевой сбор (Habr, Telegram, HeadHunter), так и полностью автономный детерминированный режим (`--dry-run`), гарантирующий мгновенную работу без интернета и VPN.

## Архитектура подсистемы (`modules/vacancies/`)
- `models.py`: Pydantic v2 модели данных (`VacancyItem`, `SearchCriteria`). Строгая типизация, валидация зарплатных границ, скора и списков навыков.
- `filter_engine.py`: Многофакторное ядро оценки (скоринг 0–100, вилки зарплат, стоп-факторы `Senior`, `Lead`, `1С программирование`).
- `sources/`: Модульные адаптеры поставщиков (`habr.py`, `telegram.py`, `hh_source.py`, `mock_source.py`).
- `fixtures/sample_vacancies.json`: 16 эталонных синтетических вакансий для автономного запуска и тестирования.
- `monitor.py`: Консольный раннер с `argparse`, поддержкой `rich`-таблиц и экспортом в Markdown/JSON.

## Параметры командной строки (`monitor.py`)
```text
python modules/vacancies/monitor.py [ПАРАМЕТРЫ]

Параметры:
  -s, --source [all|habr|telegram|hh|mock]
                        Источник вакансий (по умолчанию: all).
  -d, --direction [all|analyst|ai_automation]
                        Целевое направление (по умолчанию: all).
  -q, --query TEXT      Дополнительные ключевые слова для поискового запроса.
  -m, --min-salary INT  Минимальный порог зарплаты в рублях (по умолчанию: 50000).
  --require-salary      Отсеивать вакансии без указанной вилки (по умолчанию: False).
  --remote-only / --no-remote-only
                        Фильтровать строго удаленку / Самару (по умолчанию: True).
  -l, --limit INT       Лимит выводимых результатов (по умолчанию: 15).
  --dry-run             Автономный режим на фикстурах (гарантирует exit code 0).
  -f, --format [table|markdown|json]
                        Формат вывода данных (по умолчанию: table).
  -o, --output PATH     Путь для сохранения отчета в файл.
```

## Примеры использования
```bash
# Автономный запуск на локальных фикстурах без интернета
python modules/vacancies/monitor.py --dry-run --direction all --format table

# Поиск вакансий системного аналитика с зарплатой от 60 000 руб в JSON
python modules/vacancies/monitor.py --dry-run --direction analyst --min-salary 60000 --format json

# Мониторинг направления AI-автоматизации с сохранением в Markdown
python modules/vacancies/monitor.py --dry-run --direction ai_automation --format markdown --output storage/ai_vacancies.md
```
