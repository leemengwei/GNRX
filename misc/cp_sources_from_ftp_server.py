#！/usr/bin/python3
#-*-coding:utf-8-*-
"""
FTP常用操作
"""
from ftplib import FTP
import os
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
from sklearn.externals import joblib
import json
import shutil
import glob
from IPython import embed
import datetime


def get_data():
    dirs = glob.glob('../SUPower-Wind*')
    os.system('rm lmw_collected -r')
    
    os.system('mkdir lmw_collected/')
    for dir in dirs:
        dir = dir.split('/')[-1]
        os.system('mkdir lmw_collected/%s'%dir)
        for station in station_names.iterrows():
            print(station)
            station = station[1].values[0]
            cmd = 'cp ../%s/%s lmw_collected/%s -r'%(dir, station, dir)
            os.system(cmd)



class FTP_OP(object):
    def __init__(self, host, username, password, port):
        """
        初始化ftp
        :param host: ftp主机ip
        :param username: ftp用户名
        :param password: ftp密码
        :param port:  ftp端口 （默认21）
        """
        self.host = host
        self.username = username
        self.password = password
        self.port = port
    def ftp_connect(self):
        """
        连接ftp
        :return:
        """
        ftp = FTP()
        ftp.set_debuglevel(0)  # 不开启调试模式
        ftp.connect(host=self.host, port=self.port)  # 连接ftp
        ftp.login(self.username, self.password)  # 登录ftp
        return ftp
    def download_file(self, filenames_seeking, dst_file_path):
        """
        从ftp下载文件到本地
        :param filenames_seeking: ftp下载文件路径
        :param dst_file_path: 本地存放路径
        :return:
        """
        #buffer_size = 10240  #默认是8192
        ftp = self.ftp_connect()
        for ftp_file in filenames_seeking:
            print("Transfering %s"%ftp_file)
            write_file = dst_file_path + ftp_file
            ftp.retrbinary('RETR {0}'.format(ftp_file), open(write_file, "wb").write)
        ftp.quit()
    def generate_filenames_I_want(self, station_names_file, dst_file_path):  #from all meteor source
        print("Getting filenames")
        station_names = pd.read_csv(station_names_file, header=None).values
        ftp = self.ftp_connect()
        all_source = ftp.nlst('/SUPower-Wind*') + ftp.nlst('/MultiWindpower*')
        filenames_seeking = [] 
        filedirs_seeking = []
        for this_source in all_source:
            os.system('mkdir %s/%s'%(dst_file_path, this_source))
            for this_station in station_names:
                #print("Combining %s %s"%(this_source, this_station))
                filedir_seeking = this_source + '/' + this_station[0]
                filedirs_seeking.append(filedir_seeking)
        for this_dir in filedirs_seeking:
             WPDs = ftp.nlst(this_dir)
             if len(WPDs)>0:
                 this_source = this_dir.split('/')[-2]
                 this_code = this_dir.split('/')[-1]
                 cmd = 'mkdir %s/%s/%s'%(dst_file_path, this_source, this_code)
                 print(len(WPDs), cmd)
                 os.system(cmd)
                 filenames_seeking += WPDs
             else:
                 #print("Not found files under %s"this_dir)
                 pass
        return filenames_seeking
 

if __name__ == '__main__':
    host = "10.128.0.100"
    username = "supower"
    password = "sPRIXIN_supqx2^9"
    port = 21
    station_names_file = './stations_trading'
    dst_file_path = "/home/user/leemengwei/Grasp_data_from_web/ftp_dir/"
    ftp = FTP_OP(host=host, username=username, password=password, port=port)

    #automatically generate filenames I want:
    filenames_seeking = ftp.generate_filenames_I_want(station_names_file, dst_file_path)
    ftp.download_file(filenames_seeking=filenames_seeking, dst_file_path=dst_file_path)




