import yaml
import inspect
import sys

class ConfigValidator:
    def __init__(self, config_file):
        with open(config_file) as infile:
            self.config_dict = yaml.load(infile, Loader=yaml.SafeLoader)

        self.is_valid = True
        self.validate()

    def get_dict(self):
        if self.is_valid:
            return self.config_dict

        
    def allowed_optimizers(self):
        allowed_optimizers = [
            'adam',
            'sgd'
        ]

        return allowed_optimizers

    def allowed_loss_fnc(self):
        allowed_loss_fnc = [
            'nllloss'
        ]

        return allowed_loss_fnc

    def validate(self):
        self.validate_project()
        self.validate_data()
        self.validate_model()
        self.validate_training()
        self.validate_optimization()
        self.validate_tuning()
        self.validate_validation()

    def validate_project(self):
        project = self.config_dict['project']
        
        if not isinstance(project['name'], str):
            self.is_valid = False

        if not isinstance(project['task'], str):
            self.is_valid = False
        

    def validate_data(self):
        data = self.config_dict['data']
        
        if not isinstance(data['location'], str):
            self.is_valid = False

        if not isinstance(data['img-res'], list):
            self.is_valid = False
        else:
            if not len(data['img-res']) == 2:
                self.is_valid = False
            if not isinstance(data['img-res'][0], int) or not isinstance(data['img-res'][1], int):
                self.is_valid = False

        if not isinstance(data['greyscale'], bool):
            self.is_valid = False

    def validate_model(self):
        model = self.config_dict['model_architecture']
        architecture = model['name']

        clsmembers = [model[0] for model in inspect.getmembers(sys.modules['src.models.architectures'], inspect.isclass)]
        if architecture not in clsmembers:
            self.is_valid = False

    def validate_training(self):
        training = self.config_dict['training']

        if not isinstance(training['max_epochs'], int):
            self.is_valid = False

        if not isinstance(training['batch_size'], list):
            if not isinstance(training['batch_size'], int):
                self.is_valid = False
        else:
            for item in training['batch_size']:
                if not isinstance(item, int):
                    self.is_valid = False

    def validate_optimization(self):
        optimization = self.config_dict['optimization']

        if optimization['optimizer'].lower() not in self.allowed_optimizers():
            self.is_valid = False

        if not isinstance(optimization['learning_rate'], list):
            if not isinstance(optimization['learning_rate'], float):
                self.is_valid = False
        else:
            if not len(optimization['learning_rate']) == 2:
                self.is_valid = False
            else:
                for item in optimization['learning_rate']:
                    if not isinstance(item, float):
                        self.is_valid = False

        if optimization['loss_fnc'].lower() not in self.allowed_loss_fnc():
            self.is_valid = False
        
    def validate_tuning(self):
        tuning = self.config_dict['tuning']

        if not isinstance(tuning['number_trials'], int):
            self.is_valid = False
        
    def validate_validation(self):
        validation = self.config_dict['validation']

        if not isinstance(validation['test_size'], float):
            self.is_valid = False
        else:
            if validation['test_size'] < 0 or validation['test_size'] > 1:
                self.is_valid = False

        if not isinstance(validation['folds'], int):
            if validation['folds']:
                self.is_valid = False
