import os
import pymysql
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
from sklearn.externals import joblib
import json
import shutil
#from IPython import embed
import sys
sys.path.insert(0, os.getcwd())

'''
Reads in a station list of station names, try to connect to ZhiKong Platfrom,
grasp columns of data from it. Generating outputs:
Outputs:
  1, 测试场站总览.xlsx which consist of row of stations' info
  2, true/code.csv which is the downloaded data.
If some station is not yet contained in platform, neither in data nor csv 
will be generated, could go ask ZhangQiang, or refer to my script at
Jibei_project/src/1_preprocessing_raw_data.py generating your own.
'''


def read_statistic_base(cnName,startDate,endDate,readweather, just_info=False):
  # Search Farm Infomation
  numberDay = '1'
#  conn = pymysql.connect("10.10.10.81","test","huangyanadmin","total")
  conn = pymysql.connect("10.10.10.81","test","huangyanadmin","total")
  cursor = conn.cursor()
  sqlInfo = "select database_name,database_user,database_passwd,database_ip,database_port,type,id,run_cap,alias_name,name,forecast_dir,plant,longitude,latitude from plant_bussiness where code='"+cnName+"'"
  cursor.execute(sqlInfo)
  results = cursor.fetchall()
  plantDcID = results[0][11]
  Cap = results[0][7]
  plant_name = results[0][8]
  FarmType = results[0][5]
  # Search FarmInfo in FarmBase
  dbname = results[0][0]
  username = results[0][1]
  databasePasswd = results[0][2]
  planIdInfo = results[0][6]
  longitude = results[0][12]
  latitude = results[0][13]
  if just_info:return None,Cap,plant_name,FarmType,longitude,latitude
  conn = pymysql.connect(results[0][3],username,databasePasswd,dbname)
  cursor = conn.cursor()
  # Time Calculate
  struct_startdate = datetime.datetime.strptime(startDate, "%Y-%m-%d")
  delta = datetime.timedelta(days = 1)
  startdate_1 = struct_startdate - delta
  delta = datetime.timedelta(minutes = 5)
  starttimestr = (struct_startdate + delta).strftime('%Y-%m-%d %H:%M:%S')
  delta = datetime.timedelta(hours = 23.75)
  starttime = startdate_1 + delta
  datestartnum = starttime
  struct_enddate = datetime.datetime.strptime(endDate, "%Y-%m-%d")
  delta = datetime.timedelta(minutes = 15)
  dateendnum = struct_enddate + delta
  endtimestr = struct_enddate.strftime('%Y-%m-%d %H:%M:%S')
  alltimedatabase = [(struct_startdate+i*delta).strftime('%Y-%m-%d %H:%M:%S') for i in range(0,int((struct_enddate-struct_startdate)/delta))]
  alltimeym = [(datestartnum+i*delta).strftime('%Y%m') for i in range(0,int((dateendnum-datestartnum)/delta))]
  alltimey = [(datestartnum+i*delta).strftime('%Y') for i in range(0,int((dateendnum-datestartnum)/delta))]
  bdymstr = np.unique(alltimeym)
  allyear = np.unique(alltimey)
  print('connected')
  # Start Read Power and Weather Data
  if FarmType==0:                          #If Farm is WindFarm
      # Read dq-pre-Power Data
      cursdqalldata = tuple()
      for iy in allyear:
        dqDataCondition = 'select DATE_FORMAT(date_time,"%Y-%m-%d %H:%i:%S"),wind_speed,'\
        +'power from windforcast_'+iy+' aForcast where plant_id ='+str(planIdInfo)+' and '\
        +'aForcast.date_time> "'+starttimestr+'" and aForcast.date_time<= "'+endtimestr+'" and '\
        +'DATE_FORMAT(aForcast.forcast_date,"%k")<7 and number_days = '+numberDay+' order by date_time'
        cursor.execute(dqDataCondition)
        cursdqdata = cursor.fetchall()
        cursdqalldata = cursdqalldata+cursdqdata
      dqdata = pd.DataFrame(list(cursdqalldata))
      if len(dqdata)>0:
        dqdata.columns=['time','fore_spd','fore_power']
        dqdata.index=list(dqdata.time)
        dqdata = dqdata.apply(pd.to_numeric,errors='ignore')
        dqdata = dqdata.loc[dqdata.time.duplicated()==0]
      # Read his-pre-Power Data
      pwDataCondition = "select id from analoginput where TYPE=(select id from measurementtype "\
      +"where name='WINDSPEED') and EQUIPMENTCONTAINER_TABLEID=1071 and EQUIPMENTCONTAINER_ID="+str(plantDcID)
      cursor.execute(pwDataCondition)
      curspwdata = cursor.fetchall()
      vtrueiddata = list(curspwdata[0])[0]
      pwDataCondition = "select id from analoginput where TYPE=(select id from measurementtype "\
      +"where name='P') and EQUIPMENTCONTAINER_TABLEID=1071 and EQUIPMENTCONTAINER_ID="+str(plantDcID)
      cursor.execute(pwDataCondition)
      curspwdata = cursor.fetchall()
      ptrueiddata = list(curspwdata[0])[0]
      curstruealldata = tuple()
      for iym in bdymstr:
        try:
          vtrueDataCondition = 'select DATE_FORMAT(a.HDTIME,"%Y-%m-%d %H:%i:%S"),a.AVERAGE,a.id from hdranastat15m'\
          +iym+' a where (a.id='+str(vtrueiddata)+' or a.id='+str(ptrueiddata)+')and a.HDTIME> "'+starttimestr\
          +'" and a.HDTIME<="'+endtimestr+'" order by a.HDTIME'
          cursor.execute(vtrueDataCondition)
          curspwdata = cursor.fetchall()
          curstruealldata = curstruealldata+curspwdata
        except:
          pass
      dftruealldata = pd.DataFrame(list(curstruealldata))
      if len(dftruealldata)>0:
        dftruealldata.columns=['time','value_true','flg_true']
        v_data = dftruealldata.loc[dftruealldata["flg_true"]==vtrueiddata].sort_values(["time"])
        v_data.index = list(v_data.time)
        v_data.columns = ['time','obs_true','flg']
        v_data = v_data.loc[v_data.time.duplicated()==0]
        p_data = dftruealldata.loc[dftruealldata["flg_true"]==ptrueiddata].sort_values(["time"])
        p_data.index = list(p_data.time)
        p_data.columns = ['time','power_true','flg']
        p_data = p_data.loc[p_data.time.duplicated()==0]
      else:
        v_data = pd.DataFrame([0])
        v_data.columns=['obs_true']
        p_data = pd.DataFrame([0])
        p_data.columns=['power_true']
      # Read Weather Data
      if readweather == 1:
        #print('wind, read weather')
        cursqxalldata = tuple()
        for iy in allyear:
          qxDataCondition = 'select DATE_FORMAT(date_time,"%Y-%m-%d %H:%i:%S"),wind_speed,wind_direction'\
          +",temperature,humidity,pressure,power,qx_id  from windweather_"+iy+" aWeather  where plant_id"\
          +" = "+str(planIdInfo)+' and aWeather.date_time> "'+starttimestr+'" and aWeather.date_time<= "'\
          +endtimestr+'" and DATE_FORMAT(aWeather.weatherforcast_date,"%k")<7 and ((DATE_FORMAT(aWeather.weatherforcast_date,"%k")>1'\
          +' and aWeather.qx_id=1) or aWeather.qx_id>=2)  and number_days = '+numberDay+' order by date_time'
          cursor.execute(qxDataCondition)
          cursqxdata = cursor.fetchall()
          cursqxalldata = cursqxalldata+cursqxdata
        dfalldata = pd.DataFrame(list(cursqxalldata))
        if not dfalldata.empty:
          dfalldata.columns = ['time','spd','dir','tem','hum','pre','pow','flg']
          dfalldata.index = list(dfalldata.time)
          dfalldata = dfalldata.apply(pd.to_numeric,errors='ignore')
          qx_type = np.sort(dfalldata.flg.unique())
          dfalldata = dfalldata.loc[dfalldata.duplicated()==0]
          # merge data
          alldata = pd.DataFrame(list(alltimedatabase))
          alldata.index = alltimedatabase
          alldata.columns = ['time']
          labeldata = alldata
          if len(dqdata)>0:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_spd,dqdata.fore_power],axis=1)
          else:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
          # gs fore-qx data
          for i in qx_type:
            qx_data = dfalldata.loc[dfalldata["flg"]==i].sort_values(["time"])
            qx_data = qx_data.loc[qx_data.time.duplicated()==0]
            gs_data = pd.concat([labeldata,qx_data],axis=1).loc[:,'spd':'pow']
            gs_data.columns=['spd_'+str(i),'dir_'+str(i),'tem_'+str(i),'hum_'+str(i),'pre_'+str(i),'pow_'+str(i)]
            alldata = alldata.join(gs_data,how='right')
        else:
          # merge data
          alldata = pd.DataFrame(list(alltimedatabase))
          alldata.index = alltimedatabase
          alldata.columns = ['time']
          labeldata = alldata
          if len(dqdata)>0:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_spd,dqdata.fore_power],axis=1)
          else:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
      else:
        #print('wind, not read weather')
        # merge data
        alldata = pd.DataFrame(list(alltimedatabase))
        alldata.index = alltimedatabase
        alldata.columns = ['time']
        labeldata = alldata
        if len(dqdata)>0:
          alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_spd,dqdata.fore_power],axis=1)
        else:
          alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
        #embed()





  if FarmType==1:                        #If Farm is SolarFarm
      # Read dq-pre-Power Data
      cursdqalldata = tuple()
      for iy in allyear:
        dqDataCondition = 'select DATE_FORMAT(date_time,"%Y-%m-%d %H:%i:%S"),total_radiation,'\
        +'power from sunforcast_'+iy+' aForcast where plant_id ='+str(planIdInfo)+' and '\
        +'aForcast.date_time> "'+starttimestr+'" and aForcast.date_time<= "'+endtimestr+'" and '\
        +'DATE_FORMAT(aForcast.forcast_date,"%k")<7  and number_days = '+numberDay+' order by date_time'
        cursor.execute(dqDataCondition)
        cursdqdata = cursor.fetchall()
        cursdqalldata = cursdqalldata+cursdqdata
      dqdata = pd.DataFrame(list(cursdqalldata))
      if len(dqdata)>0:
        dqdata.columns=['time','fore_rad','fore_power']
        dqdata.index=list(dqdata.time)
        dqdata = dqdata.apply(pd.to_numeric,errors='ignore')
        dqdata = dqdata.loc[dqdata.time.duplicated()==0]
      # Read his-pre-Power Data
      pwDataCondition = "select id from analoginput where TYPE=(select id from measurementtype "\
      +"where name='IRRADIATE') and EQUIPMENTCONTAINER_TABLEID=1071 and EQUIPMENTCONTAINER_ID="+str(plantDcID)
      cursor.execute(pwDataCondition)
      curspwdata = cursor.fetchall()
      vtrueiddata = list(curspwdata[0])[0]
      pwDataCondition = "select id from analoginput where TYPE=(select id from measurementtype "\
      +"where name='P') and EQUIPMENTCONTAINER_TABLEID=1071 and EQUIPMENTCONTAINER_ID="+str(plantDcID)
      cursor.execute(pwDataCondition)
      curspwdata = cursor.fetchall()
      ptrueiddata = list(curspwdata[0])[0]
      curstruealldata = tuple()
      for iym in bdymstr:
        try:
          vtrueDataCondition = 'select DATE_FORMAT(a.HDTIME,"%Y-%m-%d %H:%i:%S"),a.AVERAGE,a.id from hdranastat15m'\
          +iym+' a where (a.id='+str(vtrueiddata)+' or a.id='+str(ptrueiddata)+')and a.HDTIME> "'+starttimestr\
          +'" and a.HDTIME<="'+endtimestr+'" order by a.HDTIME'
          cursor.execute(vtrueDataCondition)
          curspwdata = cursor.fetchall()
          curstruealldata = curstruealldata+curspwdata
        except:
          pass
      dftruealldata = pd.DataFrame(list(curstruealldata))
      if len(dftruealldata)>0:
        dftruealldata.columns=['time','value_true','flg_true']
        v_data = dftruealldata.loc[dftruealldata["flg_true"]==vtrueiddata].sort_values(["time"])
        v_data.index = list(v_data.time)
        v_data.columns = ['time','obs_true','flg']
        v_data = v_data.loc[v_data.time.duplicated()==0]
        p_data = dftruealldata.loc[dftruealldata["flg_true"]==ptrueiddata].sort_values(["time"])
        p_data.index = list(p_data.time)
        p_data.columns = ['time','power_true','flg']
        p_data = p_data.loc[p_data.time.duplicated()==0]
      else:
        v_data = pd.DataFrame([0])
        v_data.columns=['obs_true']
        p_data = pd.DataFrame([0])
        p_data.columns=['power_true']
      # Read Weather Data
      if readweather == 1:
        cursqxalldata = tuple()
        for iy in allyear:
          qxDataCondition = 'select DATE_FORMAT(date_time,"%Y-%m-%d %H:%i:%S"),total_radiation,wind_speed,wind_direction'\
          +",temperature,humidity,pressure,power,qx_id  from sunweather_"+iy+" aWeather  where plant_id"\
          +" = "+str(planIdInfo)+' and aWeather.date_time> "'+starttimestr+'" and aWeather.date_time<= "'\
          +endtimestr+'" and DATE_FORMAT(aWeather.weatherforcast_date,"%k")<7 and ((DATE_FORMAT(aWeather.weatherforcast_date,"%k")>1'\
          +' and aWeather.qx_id=1) or aWeather.qx_id>=2) and number_days = '+numberDay+' order by date_time'
          cursor.execute(qxDataCondition)
          cursqxdata = cursor.fetchall()
          cursqxalldata = cursqxalldata+cursqxdata
        dfalldata = pd.DataFrame(list(cursqxalldata))
        if not dfalldata.empty:
          dfalldata.columns = ['time','rad','spd','dir','tem','hum','pre','pow','flg']
          dfalldata.index = list(dfalldata.time)
          dfalldata = dfalldata.apply(pd.to_numeric,errors='ignore')
          qx_type = np.sort(dfalldata.flg.unique())
          dfalldata = dfalldata.loc[dfalldata.duplicated()==0]
          # merge data
          alldata = pd.DataFrame(list(alltimedatabase))
          alldata.index = alltimedatabase
          alldata.columns = ['time']
          labeldata = alldata
          if len(dqdata)>0:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_rad,dqdata.fore_power],axis=1)
          else:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
          # gs fore-qx data
          for i in qx_type:
            qx_data = dfalldata.loc[dfalldata["flg"]==1].sort_values(["time"])
            qx_data = qx_data.loc[qx_data.time.duplicated()==0]
            gs_data = pd.concat([labeldata,qx_data],axis=1).loc[:,'rad':'pow']
            gs_data.columns=['rad_'+str(i),'spd_'+str(i),'dir_'+str(i),'tem_'+str(i),'hum_'+str(i),'pre_'+str(i),'pow_'+str(i)]
            alldata = alldata.join(gs_data,how='right')
        else:
          # merge data
          alldata = pd.DataFrame(list(alltimedatabase))
          alldata.index = alltimedatabase
          alldata.columns = ['time']
          labeldata = alldata
          if len(dqdata)>0:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_rad,dqdata.fore_power],axis=1)
          else:
            alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
      else:
        # merge data
        alldata = pd.DataFrame(list(alltimedatabase))
        alldata.index = alltimedatabase
        alldata.columns = ['time']
        labeldata = alldata
        if len(dqdata)>0:
          alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true,dqdata.fore_rad,dqdata.fore_power],axis=1)
        else:
          alldata = pd.concat([labeldata,v_data.obs_true,p_data.power_true],axis=1)
  alldata.index = pd.to_datetime(alldata.index)
  # Return
  return alldata.dropna(subset=['time']),Cap,plant_name,FarmType,longitude,latitude

if __name__ == "__main__":
    import config
    station_names = pd.read_csv(config.filename, header=None)
    start_date = config.start_date 
    today = datetime.datetime.today()
    end_date = '%s-%s-%s'%(str(today.year), str(today.month), str(today.day))
    solar_station_infos = pd.DataFrame(columns=['code','lon','lat','cap','meteor_ok'])
    wind_station_infos = pd.DataFrame(columns=['code','lon','lat','cap','angle','spe','meteor_ok'])
    try:
        shutil.rmtree('./true/')
    except:
        pass
    os.mkdir('./true/')
    os.mkdir('./true/wind/')
    os.mkdir('./true/solar/')
    for station_name in station_names.iterrows():
        station_name = station_name[1].values[0]
        print("\nGrasping data from web for %s"%station_name)
        #if 1:
        try:
            alldata,Cap,plant_name,FarmType,longitude,latitude = read_statistic_base(station_name, start_date, end_date, readweather=0)
            if FarmType == 1:
                 alldata.to_csv('./true/solar/%s.csv'%station_name, index=None)
            else:
                 alldata.to_csv('./true/wind/%s.csv'%station_name, index=None)              
            print('Success for %s'%station_name)
        except:
            try:
                alldata,Cap,plant_name,FarmType,longitude,latitude = read_statistic_base(station_name, start_date, end_date, readweather=0, just_info=True)
                print("*Warning, no data and no station info will be generated for %s"%station_name, e)
                print(Cap,plant_name,FarmType,longitude,latitude)
            except Exception as e:
                print(e)
                print("*Warning, no data and no station info will be generated for %s"%station_name, e)
                continue

        if FarmType == 1:   #1 for solar
            solar_station_infos = solar_station_infos.append({'code':station_name, 'lon':longitude, 'lat':latitude, 'cap':Cap, 'angle':36, 'spe':1, 'meteor_ok':1}, ignore_index=True)
        else:  # FarmType == 'wind':
            wind_station_infos = wind_station_infos.append({'code':station_name, 'lon':longitude, 'lat':latitude, 'cap':Cap, 'angle':36, 'spe':1, 'meteor_ok':1}, ignore_index=True)

    #Form overall infos:
    if len(solar_station_infos) > 0:
        solar_station_infos[['code','lon','lat','cap','meteor_ok']].to_excel("./solar_station_infos.xlsx", index=None)
    if len(wind_station_infos) > 0:
        wind_station_infos[['code','lon','lat','cap','meteor_ok']].to_excel("./wind_station_infos.xlsx", index=None)



