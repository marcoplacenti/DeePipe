import yaml
import inspect
import sys
import logging

class Configurator:
    def __init__(self, config_file):
        with open(config_file) as infile:
            self.config_dict = yaml.load(infile, Loader=yaml.SafeLoader)

        self.data_processing_flag = False
        self.train_flag = False
        self.evaluation_flag = False
        self.deployment_flag = False

        self.is_valid = True

    def get_data_flag(self):
        return self.data_processing_flag

    def get_train_flag(self):
        return self.train_flag

    def get_evaluation_flag(self):
        return self.evaluation_flag

    def get_deployment_flag(self):
        return self.deployment_flag

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
        self.validate_deployment()

    def validate_project(self):
        project = self.config_dict['project']
        
        if not isinstance(project['name'], str):
            self.is_valid = False
            raise ValueError("The project needs to have a name. Please provide one.")

        if not isinstance(project['task'], str):
            self.is_valid = False
            raise ValueError("A task needs to be specified. "+
                "Please choose between Classification or Regression. "+
                "Currently only Classification is supported.")
        

    def validate_data(self):
        data = self.config_dict['data']
        
        if not isinstance(data['location'], str):
            self.is_valid = False
            raise ValueError("The provided path is not valid. "+
                "Please either specify a local path or an S3 path.")

        if not isinstance(data['img-res'], list):
            self.is_valid = False
            raise ValueError("Image resolution must be a list containing two integers. "+
                "Input was not a list.")
        else:
            if not len(data['img-res']) == 2:
                self.is_valid = False
                raise ValueError("Image resolution must be a list containing two integers. "+
                    "Given list does not contain two integers.")
            if not isinstance(data['img-res'][0], int) or not isinstance(data['img-res'][1], int):
                self.is_valid = False
                raise ValueError("Image resolution must be a list containing two integers. "+
                    "Either one or both the values are not integers.")

        if not isinstance(data['greyscale'], bool):
            self.is_valid = False
            raise ValueError("Invalid greyscale flag. Please provide a boolean value.")

        self.data_processing_flag = True
        

    def validate_model(self):
        model = self.config_dict['model_architecture']
        architecture = model['name']

        clsmembers = [model[0] for model in inspect.getmembers(sys.modules['deeppype.models.architectures'], inspect.isclass)]
        if architecture not in clsmembers:
            self.is_valid = False
            raise ValueError("Provided model architecture name is not supported. "+
                "Please specifiy the name of an existing one or implement the desired architecture.")

    def validate_training(self):
        training = self.config_dict['training']

        if not isinstance(training['max_epochs'], int):
            self.is_valid = False
            raise ValueError("The provided value of max_epochs is not valid. Please provide an integer positive value.")

        if not isinstance(training['batch_size'], list):
            if not isinstance(training['batch_size'], int):
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")
        else:
            for item in training['batch_size']:
                if not isinstance(item, int):
                    self.is_valid = False
                    raise ValueError("ERROR MESSAGE")

        self.train_flag = True
        

    def validate_optimization(self):
        optimization = self.config_dict['optimization']

        if optimization['optimizer'].lower() not in self.allowed_optimizers():
            self.is_valid = False
            raise ValueError("ERROR MESSAGE")

        if not isinstance(optimization['learning_rate'], list):
            if not isinstance(optimization['learning_rate'], float):
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")
        else:
            if not len(optimization['learning_rate']) == 2:
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")
            else:
                for item in optimization['learning_rate']:
                    if not isinstance(item, float):
                        self.is_valid = False
                        raise ValueError("ERROR MESSAGE")

        if optimization['loss_fnc'].lower() not in self.allowed_loss_fnc():
            self.is_valid = False
            raise ValueError("ERROR MESSAGE")
        
    def validate_tuning(self):
        tuning = self.config_dict['tuning']

        if not isinstance(tuning['number_trials'], int):
            self.is_valid = False
            raise ValueError("ERROR MESSAGE")
        
    def validate_validation(self):
        validation = self.config_dict['validation']

        if not isinstance(validation['test_size'], float):
            self.is_valid = False
            raise ValueError("ERROR MESSAGE")
        else:
            if validation['test_size'] < 0 or validation['test_size'] > 1:
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")

        if not isinstance(validation['folds'], int):
            if validation['folds']:
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")

        self.evaluation_flag = True
        

    def validate_deployment(self):
        try:
            deployment = self.config_dict['deployment']

            if not isinstance(deployment['min_test_score'], float):
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")
            else:
                if deployment['min_test_score'] < 0 or deployment['min_test_score'] > 1:
                    self.is_valid = False
                    raise ValueError("ERROR MESSAGE")

            if not isinstance(deployment['endpoint_name'], str):
                if deployment['endpoint_name']:
                    self.is_valid = False
                    raise ValueError("ERROR MESSAGE")

            if not isinstance(deployment['instance_type'], str):
                self.is_valid = False
                raise ValueError("ERROR MESSAGE")

        except Exception as e:
            logging.critical(e)

        self.deployment_flag = True
                