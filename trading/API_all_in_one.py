import dependency_misc
import dependency_data_labour
import trading_agent_trainer
import trading_agent_deployer
from IPython import embed

def mengxi_deployment(req_json):
    # config 
    config = dependency_misc.arg_parse() 

    # data is also given by req
    raw_data = dependency_data_labour.mengxi_raw_data_prepare(req_json)
    
    # run over circumstances
    if req_json['mode'].upper() == 'TRAIN':
        Trainer = trading_agent_trainer.ModelTrainer(req_json, raw_data)
        Trainer.run()
        return Trainer.rep_json
    else:
        raw_data = raw_data[req_json['wanted_day']]
        Runner = trading_agent_deployer.ModelRunner(req_json, raw_data)
        Runner.run()
        return Runner.rep_json


if __name__ == '__main__':
    # Get requests json:
    req_json_train = dependency_misc.get_req('train')
    req_json_deploy = dependency_misc.get_req('deploy')
    
    # Get responses json:
    rep_json_train = mengxi_deployment(req_json_train)
    rep_json_deploy = mengxi_deployment(req_json_deploy)


