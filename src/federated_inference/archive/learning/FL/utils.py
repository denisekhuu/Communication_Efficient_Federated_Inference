from numpy.random import default_rng
class ModelAggregator():

    @staticmethod
    def model_avg(parameters):
        new_params = {}
        for name in parameters[0].keys():
            new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)
        return new_params


class ClientSelector():

    @staticmethod
    def random_selector(number_of_clients, clients_per_round):
        rng = default_rng()
        return rng.choice(number_of_clients, size=clients_per_round, replace=False)
