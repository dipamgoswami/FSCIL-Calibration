def get_model(model_name, args):
    name = model_name.lower()
    if name == "adam_adapter":
        from models.adam_adapter import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    else:
        assert 0
    
    return Learner(args)
