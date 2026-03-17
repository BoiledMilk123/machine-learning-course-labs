# lab7/Genetic.py
import random
import matplotlib.pyplot as plt
import numpy as np

from VisualizerGraph import *
from Group import *




class Genetic:
    #популяция
    def __init__(self):
        self.population = []

    @staticmethod
    def create_individual():
        return Group([random.randint(0, NUM_FACTORIES - 1) for _ in range(TOTAL_LOCATIONS)])

    def create_population(self, n=0):
        self.population = [self.create_individual() for _ in range(n)]

    # распределение выпуска заводов между городами (по назначению в хромосоме)
    @staticmethod
    def distribute_supply(individual):
        # delivered[j] — сколько получил город j
        delivered = [0.0 for _ in range(TOTAL_LOCATIONS)]
        # транспортные расходы
        total_transport_cost = 0.0
        # выпуск заводов, которые никому не назначены
        unused_supply = 0.0

        # собираем список городов для каждого завода
        assigned = [[] for _ in range(NUM_FACTORIES)]
        for city_index in range(TOTAL_LOCATIONS):
            factory_index = individual[city_index]
            assigned[factory_index].append(city_index)

        # распределяем выпуск каждого завода между назначенными городами
        for factory_index in range(NUM_FACTORIES):
            cities = assigned[factory_index]
            supply = PRODUCTION_CREATION[factory_index]

            if len(cities) == 0:
                unused_supply += supply
                continue

            total_demand = sum(CITY_DEMAND[c] for c in cities)
            if total_demand <= 0:
                unused_supply += supply
                continue

            # сколько реально можно/нужно отгрузить: не превышаем суммарный спрос
            ship_total = min(supply, total_demand)
            # неиспользованный выпуск (если производство больше спроса назначенных городов)
            unused_supply += (supply - ship_total)

            # делим ship_total пропорционально спросу городов
            for c in cities:
                shipped = ship_total * (CITY_DEMAND[c] / total_demand)
                delivered[c] += shipped
                # стоимость = сколько везём * цена за единицу (расстояние)
                total_transport_cost += shipped * OIL_COSTS[factory_index][c]

        return delivered, total_transport_cost, unused_supply

    @staticmethod
    def calculate_fitness(individual) -> float:
        # недопоставка
        total_under = 0.0
        # перепоставка
        total_over = 0.0
        # транспортные расходы
        total_transport_cost = 0.0
        # неиспользованный выпуск
        unused_supply = 0.0

        delivered, total_transport_cost, unused_supply = Genetic.distribute_supply(individual)

        for city_index in range(TOTAL_LOCATIONS):
            demand = CITY_DEMAND[city_index]
            got = delivered[city_index]

            if got < demand:
                total_under += (demand - got)
            else:
                total_over += (got - demand)

        # ограничение по бюджету (если задано)
        budget_penalty = 0.0
        if MAX_TRANSPORT_COST is not None and total_transport_cost > MAX_TRANSPORT_COST:
            budget_penalty = total_transport_cost - MAX_TRANSPORT_COST

        # цель: минимизировать транспорт и штрафы
        objective = (
            total_transport_cost
            + W_UNDER * total_under
            + W_OVER * total_over
            + W_UNUSED * unused_supply
            + 10.0 * budget_penalty
        )

        #приспособленность зависит от штрафов и транспортных расходов
        fitness = 1 / (1 + objective) * 100
        return fitness

    def genetic_algorithm(self):
        self.create_population(n=POP_SIZE)
        #счетчик
        generation = 0
        #статистика
        max_fitness_values = []
        mean_fitness_values = []

        while generation < GEN_LIMIT:
            #приспособленность
            fitness_scores = [self.calculate_fitness(ind) for ind in self.population]
            for individual, fitness_score in zip(self.population, fitness_scores):
                individual.fitness = fitness_score

            max_fitness = max(fitness_scores)
            mean_fitness = sum(fitness_scores) / len(self.population)

            max_fitness_values.append(max_fitness)
            mean_fitness_values.append(mean_fitness)

            #вывод статистики
            print(f'Поколение {generation + 1}: Макс. приспособ. = {max_fitness}, Ср. приспособ. = {mean_fitness}')
            best_index = fitness_scores.index(max(fitness_scores))
            best_ind = np.array(self.population[best_index]) + 1
            print("Лучший индивидуум = ", *best_ind, "\n")

            generation += 1
            #список выбранных индивидов в ходе отбора
            #selected = self.tournament_selection()
            selected = self.rank_selection()

            #создание потомков
            offspring = [self.clone_individual(ind) for ind in selected]

            # берем 0 и 1 элементы списка
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    # self.crossover_one_point(child1, child2)
                    #self.crossover_two_points(child1, child2)
                    self.crossover_uniform(child1, child2)

            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    #self.mutate_flip(mutant)
                     #self.mutate_swap(mutant)
                     self.mutate_scramble(mutant)

            self.population = offspring

        # Визуализация результатов
        plt.figure(figsize=(10, 6))
        plt.plot(max_fitness_values, label='Максимальная приспособленность', color="red")
        plt.plot(mean_fitness_values, label='Средняя приспособленность', color="blue")
        plt.xlabel('Поколение')
        plt.ylabel('Приспособленность')
        plt.title("Изменение приспособленности по поколениям")
        plt.legend()
        plt.grid()
        plt.show()

        # пересчитать fitness для финальной популяции (последнего поколения)
        fitness_scores = [self.calculate_fitness(ind) for ind in self.population]
        for individual, fitness_score in zip(self.population, fitness_scores):
            individual.fitness = fitness_score

        best_overall = max(self.population, key=lambda ind: ind.fitness)

        # печать итоговых метрик (удобно для отчёта)
        delivered, total_transport_cost, unused_supply = self.distribute_supply(best_overall)
        total_under = 0.0
        total_over = 0.0
        for city_index in range(TOTAL_LOCATIONS):
            demand = CITY_DEMAND[city_index]
            got = delivered[city_index]
            if got < demand:
                total_under += (demand - got)
            else:
                total_over += (got - demand)

        print("ИТОГО:")
        print("Транспортные расходы =", round(total_transport_cost, 3))
        print("Недопоставка =", round(total_under, 3))
        print("Перепоставка =", round(total_over, 3))
        print("Неиспользованный выпуск =", round(unused_supply, 3))
        print("Решение (город->завод) =", [x + 1 for x in best_overall], "\n")

        VisualizerGraph.visualize_best_solution(best_overall)
        VisualizerGraph.visualize_oil_costs()

    @staticmethod
    def clone_individual(ind):
        clone = Group(ind[:])
        clone.fitness = ind.fitness
        return clone

    #отбор
    def tournament_selection(self):
        selected = []
        for _ in range(len(self.population)):
            i1, i2, i3 = random.sample(range(len(self.population)), 3)
            selected.append(max([self.population[i1], self.population[i2], self.population[i3]], key=lambda x: x.fitness))
        return selected

    def rank_selection(self):
        ranked_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        #сумма рангов
        total_rank = sum(range(len(ranked_population)))
        selected = []
        for _ in range(len(self.population)):
            pick = random.randint(0, total_rank)
            current = 0
            for i, ind in enumerate(ranked_population):
                current += (len(ranked_population) - i)
                if current > pick:
                    selected.append(ind)
                    break
        return selected

    # Кроссовер
    @staticmethod
    def crossover_one_point(child1, child2):  #обмен генов от точки
        point = random.randint(1, len(child1) - 2)
        child1[point:], child2[point:] = child2[point:], child1[point:]

    @staticmethod
    def crossover_two_points(child1, child2):  #обмен генов между 2 точками
        point1 = random.randint(1, len(child1) - 2)
        point2 = random.randint(point1 + 1, len(child1) - 1)
        child1[point1:point2], child2[point1:point2] = child2[point1:point2], child1[point1:point2]

    @staticmethod
    def crossover_uniform(child1, child2):  #случайный обмен генов с вероятностью
        for i in range(len(child1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]

    # Мутации
    @staticmethod
    def mutate_flip(mutant, rate=0.02):
        for idx in range(len(mutant)):
            if random.random() < rate:
                new_val = mutant[idx]
                while new_val == mutant[idx]:
                    new_val = random.randint(0, NUM_FACTORIES - 1)
                mutant[idx] = new_val

    @staticmethod
    def mutate_swap(mutant):
        idx1, idx2 = random.sample(range(len(mutant)), 2)
        mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]

    @staticmethod
    def mutate_scramble(mutant):
        start_idx = random.randint(0, len(mutant) - 1)
        end_idx = random.randint(start_idx + 1, len(mutant))
        segment = mutant[start_idx:end_idx]
        random.shuffle(segment)
        mutant[start_idx:end_idx] = segment
