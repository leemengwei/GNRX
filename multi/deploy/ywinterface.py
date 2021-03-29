import sys
import os
curdir="/home/sprixin/fcstalgorithm/lmwfcst1"

sys.path.append(curdir)
import api_by_lmw

print("Main calling API")

class OptInterface():
    def OPT_Module(self, json):
        return  api_by_lmw.main(json)


if __name__ == '__main__':
    request_json =  \
	{ \
	"algopath": "/home/sprixin/pytest/create/ywinterface.py", \
	#'fcstdt': '2021-01-06 18:07:50', \
	'fcstdt': '2020-12-23 07:28:19', \
	"plants": 
		[ \
#          {'farmId': 730, 'code': 'SXGDGJB', 'modelId': 107, 'daynum': 10, 'runcap': 198.0, 'limitBegin': '2020-08-01 00:00:00', 'limitEnd': '2030-09-01 00:00:00', 'limitFactor': 0.8999999761581421, 'expandBegin': '1970-01-01 08:00:00', 'expandEnd': '1970-01-01 08:00:00', 'expandFactor': 1.0, 'modeldir': '/home/user/leemengwei/y业务平台接口/deploy/data/SXGDGJB/', 'qxlist': [{'qxid': 7, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/7/0/SXGDGJB'}, {'qxid': 9, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/9/0/SXGDGJB'}, {'qxid': 17, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/17/0/SXGDGJB'}], 'outfcstfile': '../output/SXGDGJB.txt'}, #no P
          {'farmId': 730, 'code': 'SXGDGJB', 'modelId': 107, 'daynum': 10, 'runcap': 198.0, 'limitBegin': '2020-08-01 00:00:00', 'limitEnd': '2030-09-01 00:00:00', 'limitFactor': 0.8999999761581421, 'expandBegin': '1970-01-01 08:00:00', 'expandEnd': '1970-01-01 08:00:00', 'expandFactor': 1.0, 'modeldir': '/home/user/leemengwei/y业务平台接口/deploy/data/SXGDGJB/', 'qxlist': [{'qxid': 10, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/10/0/SXGDGJB'}, {'qxid': 12, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/12/0/SXGDGJB'}, {'qxid': 19, 'qxcode': 'SXGDGJB', 'qxdir': '/home/user/leemengwei/y业务平台接口/deploy/test/2021-0318-多源混合模型新增气象预测结果/weather/19/0/SXGDGJB'}], 'outfcstfile': '../output/SXGDGJB.txt'}, #P
#          {
#           "farmId": 730, \
#           "code": "GSZGHHSG", \
#           #"code": "SXTRJXSQ01",
#           "modelId": 107, \
#           "daynum": 6, \
#           "runcap": 400.0, \
#           "limitBegin": "2020-08-01 00:00:00", \
#           "limitEnd": "2030-01-01 15:00:00", \
#           "limitFactor": 1.0, \
#           "expandBegin": "1970-01-01 08:00:00", \
#           "expandEnd": "1970-01-01 08:00:00", \
#           "expandFactor": 1, 
#           "modeldir": "../data/GSZGHHSG/", \
#           "qxlist": [ \
#          	{
#          	 "qxid": 7, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-7"
#          	}, 
#          	{
#          	 "qxid": 9, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-9"
#          	}, 
#          	{
#          	 "qxid": 10, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-10"
#          	}, 
#          	{
#          	 "qxid": 12, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-12"
#          	},
#          	{
#          	 "qxid": 106, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-106"
#          	}
#          	], \
#           "outfcstfile": "../output/GSZGHHSG.txt"
#          },
#          {
#           "farmId": 730, \
#           "code": "GSZGHHSG", \
#           "modelId": 107, \
#           "daynum": 10, \
#           "runcap": 400.0, \
#           "limitBegin": "2020-08-01 00:00:00", \
#           "limitEnd": "2030-01-01 15:00:00", \
#           "limitFactor": 1.0, \
#           "expandBegin": "1970-01-01 08:00:00", \
#           "expandEnd": "1970-01-01 08:00:00", \
#           "expandFactor": 1, 
#           "modeldir": "../data/GSZGHHSG/", \
#           "qxlist": [ \
#          	{
#          	 "qxid": 7, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-7"
#          	}, 
#          	{
#          	 "qxid": 9, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-9"
#          	}, 
#          	{
#          	 "qxid": 10, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-10"
#          	}, 
#          	{
#          	 "qxid": 12, \
#          	 "qxcode": "GSZGHHSG", \
#          	 "qxdir": "../data/GSZGHHSG-12"
#          	}
#          	], \
#           "outfcstfile": "../output/GSZGHHSG2.txt"
#          }
		]
	}


    opt = OptInterface()
    output = opt.OPT_Module(request_json)
    print('\n\nOutput for C', output)



