import pandas as pd
import math
import json
import os

def crowd_density(people_data, area_data):

    for i in range(len(people_data)):
        park_id = people_data[i][0]
        park_date = people_data[i][1]
        park_people = people_data[i][2]
        print("사람 수 %i명" %park_people)
        park = area_data.loc[area_data['park_id'] == park_id]
        park_area = park.iloc[0,1]
        print("면적 %fm^2" %park_area)
        park_density = ((park_people * 4 * 3.141592 * 100) / park_area)
        print("인구밀집도 %f%%" %park_density)
        park_avg_distance = math.sqrt(park_area / park_people)
        print("평균 거리 %fm\n" %park_avg_distance)
        
        # 각 공원 json 파일 쓰기
        filename = "output/park_json/" + park_id + ".json"
        file_path = filename

        json_data = {}

        with open(file_path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        
        json_data['data'].append(
            {"date": park_date, "density": park_density, "average_distance": park_avg_distance})

        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile, ensure_ascii=False, indent=4)

        # get_parks_info 쓰기
        if i == 0:
            init_parks_data = {'parks':[]}
            
            with open("/home/login/park_json/get_parks_info.json", "w") as init_parks_file:
                json.dump(init_parks_data, init_parks_file, ensure_ascii=False, indent=4) 
            
        parks_data = {}
        
        with open("/home/login/park_json/get_parks_info.json", "r", encoding="utf-8") as parks_file:
            parks_data = json.load(parks_file)
        
        with open(file_path, "r", encoding="utf-8") as park_file:
            park_data = json.load(park_file)
            
        parks_data['parks'].append(park_data)
            
        with open("/home/login/park_json/get_parks_info.json", "w") as parks_file:
            json.dump(parks_data, parks_file, ensure_ascii=False, indent=4)