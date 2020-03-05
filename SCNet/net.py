from keras.engine.saving import model_from_json

class Deep_Net():
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        pass

    def layer(self):
        pass

    def predict(self, X):
        pass

    def evaulate(self, X, y):
        pass

    def save(self, save_path, type='json'):
        if type == 'json':
            # save as JSON
            json_string = self.model.to_json()
            open(save_path + ".json", 'w').write(json_string)
            self.model.save_weights(save_path + ".h5")
        else:
            self.model.save(save_path + ".h5")
    def data_save_roc(self, x, y, pre_score, save_path, average="macro"):
        pass

    def load(self, model_path, type='json'):
        if type == 'json':
            # save as JSON
            self.model = model_from_json(open(model_path + ".json").read())
            # model_main.trainable = False
            self.model.load_weights(model_path + ".h5")
        else:
            self.model.load_model(model_path + ".h5")