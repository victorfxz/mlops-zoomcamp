class ModelMaker:
    def __init__(self,model):
        self.model = model

    def prepare_features(self,ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self,features):
        pred = self.model.predict(features)
        return float(pred[0])