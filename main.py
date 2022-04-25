from model import Model
from eda import exploratory_data_analysis
if __name__ == '__main__':

    data = exploratory_data_analysis()
    print(data.eda())
    model = Model()
    model.model()
    print(model.test_predict())
    print(model.train_predict())