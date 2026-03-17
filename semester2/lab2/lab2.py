import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl




# "Оценка риска для здоровья человека"



# BMI (индекс массы тела). нечёткие множества
bmi = ctrl.Antecedent(np.arange(10, 46, 1), 'bmi')

# Уровень физической активности по шкале 0..10
activity = ctrl.Antecedent(np.arange(0, 11, 1), 'activity')

# Выходная переменная: риск для здоровья (0..100)
health_risk = ctrl.Consequent(np.arange(0, 101, 1), 'health_risk')

# Недостаточный вес. трапеция. функции принадлежности для каждого терма
bmi['underweight'] = fuzz.trapmf(bmi.universe, [10, 10, 16, 18.5])

# Нормальный вес. треуг
bmi['normal'] = fuzz.trimf(bmi.universe, [18, 21.75, 25])

# Избыточный вес
bmi['overweight'] = fuzz.trimf(bmi.universe, [24, 27.5, 30])

# Ожирение
bmi['obesity'] = fuzz.trapmf(bmi.universe, [29, 32, 45, 45])


# Функции принадлежности для уровня активности

# Малоподвижный
activity['sedentary'] = fuzz.trimf(activity.universe, [0, 0, 3])

# Умеренный
activity['moderate'] = fuzz.trimf(activity.universe, [2, 4, 6])

# Активный
activity['active'] = fuzz.trimf(activity.universe, [5, 7, 9])

# Очень активный
activity['very_active'] = fuzz.trapmf(activity.universe, [8, 9, 10, 10])


#  для риска здоровья

# Низкий риск
health_risk['low'] = fuzz.trimf(health_risk.universe, [0, 15, 30])

# Средний риск
health_risk['medium'] = fuzz.trimf(health_risk.universe, [25, 45, 65])

# Высокий риск
health_risk['high'] = fuzz.trimf(health_risk.universe, [60, 75, 90])

# Очень высокий риск
health_risk['very_high'] = fuzz.trapmf(health_risk.universe, [85, 95, 100, 100])



# нечёткие правила

# Недостаточный вес
#В нечёткой логике всё мягче: bmi может принадлежать underweight не на 100%, а, например, на 0.6;
#activity может принадлежать sedentary на 0.8.
#Тогда сила срабатывания правила будет вычисляться по этим степеням принадлежности.
#если: первое условие истинно на 0.6, второе — на 0.8, то правило сработает примерно на уровне 0.6.
rule1 = ctrl.Rule(bmi['underweight'] & activity['sedentary'], health_risk['medium'])
rule2 = ctrl.Rule(bmi['underweight'] & activity['moderate'], health_risk['medium'])
rule3 = ctrl.Rule(bmi['underweight'] & activity['active'], health_risk['medium'])
rule4 = ctrl.Rule(bmi['underweight'] & activity['very_active'], health_risk['high'])

# Нормальный вес
rule5 = ctrl.Rule(bmi['normal'] & activity['sedentary'], health_risk['medium'])
rule6 = ctrl.Rule(bmi['normal'] & activity['moderate'], health_risk['low'])
rule7 = ctrl.Rule(bmi['normal'] & activity['active'], health_risk['low'])
rule8 = ctrl.Rule(bmi['normal'] & activity['very_active'], health_risk['low'])

# Избыточный вес
rule9 = ctrl.Rule(bmi['overweight'] & activity['sedentary'], health_risk['high'])
rule10 = ctrl.Rule(bmi['overweight'] & activity['moderate'], health_risk['medium'])
rule11 = ctrl.Rule(bmi['overweight'] & activity['active'], health_risk['medium'])
rule12 = ctrl.Rule(bmi['overweight'] & activity['very_active'], health_risk['low'])

# Ожирение
rule13 = ctrl.Rule(bmi['obesity'] & activity['sedentary'], health_risk['very_high'])
rule14 = ctrl.Rule(bmi['obesity'] & activity['moderate'], health_risk['high'])
rule15 = ctrl.Rule(bmi['obesity'] & activity['active'], health_risk['high'])
rule16 = ctrl.Rule(bmi['obesity'] & activity['very_active'], health_risk['medium'])



# СОЗДАНИЕ СИСТЕМЫ


health_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4,
    rule5, rule6, rule7, rule8,
    rule9, rule10, rule11, rule12,
    rule13, rule14, rule15, rule16
])


# ШАГ 5. ФУНКЦИЯ СИМУЛЯЦИИ

def interpret_activity(value):
    if value <= 2.5:
        return "малоподвижный"
    elif value <= 5:
        return "умеренный"
    elif value <= 8:
        return "активный"
    return "очень активный"


def interpret_risk(score):
    if score < 30:
        return "низкий"
    elif score < 60:
        return "средний"
    elif score < 85:
        return "высокий"
    return "очень высокий"


def simulate_health_system(bmi_value, activity_value):

    # Создаем отдельный объект симуляции
    health_sim = ctrl.ControlSystemSimulation(health_ctrl)

    # Передаем входные данные в систему
    health_sim.input['bmi'] = bmi_value
    health_sim.input['activity'] = activity_value

    # Запускаем нечеткий вывод
    health_sim.compute()

    # Получаем результат
    result = health_sim.output['health_risk']

    # Выводим результаты
    print("РЕЗУЛЬТАТ НЕЧЕТКОЙ СИСТЕМЫ")
    print(f"BMI: {bmi_value:.2f}")
    print(f"Уровень активности (0..10): {activity_value:.2f}")
    print(f"Категория активности: {interpret_activity(activity_value)}")
    print(f"Оценка риска для здоровья: {result:.2f} из 100")
    print(f"Итоговая категория риска: {interpret_risk(result)}")

    # Визуализация результата
    health_risk.view(sim=health_sim)




# Дополнение нечеткого множества с треугольной функцией принадлежности


def triangular_membership(x, a, b, c):

    if not (a < b < c):
        raise ValueError("Параметры должны удовлетворять условию a < b < c")

    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif x == b:
        return 1.0
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0


def complement_fuzzy_set(a, b, c, crisp_objects):
    results = []

    for x in crisp_objects:
        mu_a = triangular_membership(x, a, b, c)
        mu_not_a = 1 - mu_a
        results.append((x, mu_a, mu_not_a))

    return results


def run_complement_task():
    print("\nДОПОЛНЕНИЕ НЕЧЕТКОГО МНОЖЕСТВА")
    print("Введите параметры треугольной функции принадлежности a b c")
    print("Например: 18 22 26")
    a, b, c = map(float, input("a b c: ").split())

    print("Введите четкие объекты множества через пробел")
    print("Например: 16 18 20 22 24 26 28")
    crisp_objects = list(map(float, input("x: ").split()))

    results = complement_fuzzy_set(a, b, c, crisp_objects)

    print("\nРезультат:")
    print(" x\tmu_A(x)\tmu_not_A(x)")
    for x, mu_a, mu_not_a in results:
        print(f"{x:.2f}\t{mu_a:.3f}\t{mu_not_a:.3f}")

    # Построение графика
    x_min = min(min(crisp_objects), a) - 2
    x_max = max(max(crisp_objects), c) + 2
    x_plot = np.linspace(x_min, x_max, 400)

    mu_a_plot = [triangular_membership(x, a, b, c) for x in x_plot]
    mu_not_a_plot = [1 - mu for mu in mu_a_plot]

    plt.figure(figsize=(9, 4))
    plt.plot(x_plot, mu_a_plot, label='A(x) - исходное множество')
    plt.plot(x_plot, mu_not_a_plot, label='not A(x) - дополнение')
    plt.scatter([r[0] for r in results], [r[1] for r in results], marker='o')
    plt.scatter([r[0] for r in results], [r[2] for r in results], marker='x')

    plt.ylim(-0.05, 1.05)
    plt.xlabel('x')
    plt.ylabel('Степень принадлежности')
    plt.title('Дополнение нечеткого множества')
    plt.grid(True)
    plt.legend()



if __name__ == '__main__':



    # По желанию пользователя запускаем мини-задание
    answer = input("\nЗапустить мини-задание по дополнению нечеткого множества? (y/n): ").strip().lower()
    if answer == 'y':
        run_complement_task()



    # Отображаем графики
    plt.show()

    # Пример для медицинской системы:
    # BMI = 31 -> ближе к ожирению
    # Активность = 2 -> малоподвижный образ жизни
    simulate_health_system(bmi_value=31, activity_value=2)