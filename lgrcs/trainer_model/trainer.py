from my_tools import User, ModelHandler
from trainer_model.collect_data import collect_data


def trainer():
    print('Welcome to L.G.R.C.S training system!')
    user_name = input('Please enter your full name and press enter: ')

    user = User(user_name)
    user = collect_data(user)
    user.export_collected_data(include_pickle=True)

    model = ModelHandler(user)
    model.fit_predict()
    model.export_model()

    print('[INFO] finish.')


