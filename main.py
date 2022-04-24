from model import Model
if __name__ == '__main__':
    model = Model()
    model.model()
    print(model.test_predict())
    print(model.train_predict())