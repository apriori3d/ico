#!/usr/bin/env python3
"""Демонстрация проблемы с pickle и локальными функциями."""

import pickle
from functools import partial


def global_function():
    """Глобальная функция - может быть запиклена."""
    return "I'm global!"


def create_local_function():
    """Создает локальную функцию - НЕ может быть запиклена."""
    x = 42
    
    def local_function():
        return f"I'm local with x={x}"
    
    return local_function


def create_partial_function():
    """Создает partial функцию - МОЖЕТ быть запиклена."""
    
    def worker_func(value):
        return f"Working with {value}"
    
    return partial(worker_func, "some_value")


# Демонстрация
if __name__ == "__main__":
    print("=== Тестирование pickle с разными типами функций ===\n")
    
    # 1. Глобальная функция - работает
    print("1. Глобальная функция:")
    try:
        pickled = pickle.dumps(global_function)
        unpickled = pickle.loads(pickled)
        print(f"   ✓ Успешно: {unpickled()}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # 2. Локальная функция - НЕ работает
    print("\n2. Локальная функция:")
    local_func = create_local_function()
    try:
        pickled = pickle.dumps(local_func)
        unpickled = pickle.loads(pickled)
        print(f"   ✓ Успешно: {unpickled()}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    # 3. Partial функция - работает
    print("\n3. Partial функция:")
    partial_func = create_partial_function()
    try:
        pickled = pickle.dumps(partial_func)
        unpickled = pickle.loads(pickled)
        print(f"   ✓ Успешно: {unpickled()}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    
    print("\n=== Объяснение ===")
    print("Локальные функции не могут быть запиклены, потому что:")
    print("- Pickle ищет функции по имени в глобальном пространстве имен")
    print("- Локальные функции существуют только в области видимости метода")
    print("- При попытке восстановления pickle не может найти функцию")
    print("\nРешения:")
    print("- Использовать functools.partial с глобальной/статической функцией")
    print("- Сделать функцию глобальной или статическим методом класса")
    print("- Использовать классы вместо замыканий")