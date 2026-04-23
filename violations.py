"""Чистая бизнес-логика определения нарушителя.

Здесь нет I/O, модели YOLO и БД — только правила, поэтому модуль полностью
покрывается unit-тестами без внешних зависимостей.
"""
from typing import Literal

DetectionLabel = Literal["helmet", "no_helmet", "unknown"]


def classify_label(cls_name: str) -> DetectionLabel:
    """Приводит имя класса модели к одной из трёх внутренних меток.

    В датасете встречаются варианты 'Helmet', 'NoHelmet', 'without_helmet' и т.п.
    Сначала проверяем признаки «нет шлема», иначе слово ``without_helmet``
    ошибочно попадёт в helmet-ветку (оно содержит 'helmet', но не содержит 'no').
    """
    name = cls_name.lower()
    if "without" in name or ("no" in name and "helmet" in name):
        return "no_helmet"
    if "helmet" in name:
        return "helmet"
    return "unknown"


def no_helmet_ratio(helmet: int, no_helmet: int) -> float:
    total = helmet + no_helmet
    if total == 0:
        return 0.0
    return no_helmet / total


def should_flag_violator(
    helmet: int,
    no_helmet: int,
    *,
    min_observations: int,
    threshold: float,
) -> bool:
    """Возвращает True, если трек можно считать нарушителем.

    Требует накопить не менее `min_observations` кадров — это фильтрует короткие
    треки с ложными срабатываниями. Порог строгий: `ratio > threshold`.
    """
    total = helmet + no_helmet
    if total < min_observations:
        return False
    return no_helmet_ratio(helmet, no_helmet) > threshold
