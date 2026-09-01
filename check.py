import tensorflow as tf

# Вывод всех доступных атрибутов и модулей в TensorFlow
modules = [module for module in dir(tf) if not module.startswith('_')]

print("Список встроенных модулей в TensorFlow:")
for module in modules:
    print(module)
