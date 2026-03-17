import random
import math



POP_SIZE = 300
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
GEN_LIMIT = 100
SEED = 10

random.seed(SEED)

NUM_FACTORIES = 4
TOTAL_LOCATIONS = 10

PLANE_SIZE = 100
CAP_RANGE=(60,140)
DEMAND_RANGE = (10,35)

W_UNDER = 100.0
W_OVER = 5.0
W_UNUSED = 2.0

MAX_TRANSPORT_COST = None

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

FACTORY_XY = [(random.uniform(0, PLANE_SIZE), random.uniform(0, PLANE_SIZE)) for _ in range(NUM_FACTORIES)]
CITY_XY = [(random.uniform(0,PLANE_SIZE), random.uniform(0, PLANE_SIZE)) for _ in range(TOTAL_LOCATIONS)]

PRODUCTION_CREATION = [random.randint(*CAP_RANGE) for _ in range(NUM_FACTORIES)]
CITY_DEMAND = [random.randint(*DEMAND_RANGE) for _ in range(TOTAL_LOCATIONS)]

OIL_COSTS = [[euclid(FACTORY_XY[i], CITY_XY[j]) for j in range(TOTAL_LOCATIONS)] for i in range(NUM_FACTORIES)]

