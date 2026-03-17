class Group(list):
    fitness: float
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = 0
