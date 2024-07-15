from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import geopandas as gpd
import folium
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from sqlalchemy import create_engine
from geopy.distance import distance
from shapely.geometry import Point, box, Polygon, mapping, LineString
from pyproj import Geod
import psycopg2

USED_BUS_LINES = ['100', '108', '232', '2336', '2803', '292', '298', '3', '309', '315', '324', '328', '343', '355', '371', '388', 
                  '397', '399', '415', '422', '457', '483', '497', '550', '553', '554', '557', '565', '606', '624', '629', '634', 
                  '638', '639', '665', '756', '759', '774', '779', '803', '838', '852', '864', '867', '878', '905', '917', '918'] # SELECT DISTINCT FROM

SEGUNDA_FEIRA = ['20240429', '20240506']
TERCA_FEIRA = ['20240430', '20240507']
QUARTA_FEIRA = ['20240501', '20240508']
QUINTA_FEIRA = ['20240425', '20240502', '20240509']
SEXTA_FEIRA = ['20240426', '20240503', '20240510']
SABADO = ['20240427', '20240504']
DOMINGO = ['20240428', '20240505']

database_uri = 'postgresql://postgres:admin@localhost:5432/gps_onibus_rj'
db_engine_alchemy = create_engine(database_uri)

def database_query(linha, dates: list, engine) -> pd.DataFrame:
    queries = []
    for date in dates:
        query = (f'SELECT ordem, latitude, longitude, datahora_ts, velocidade ' 
                 f'FROM dados_gps_{date} '
                 f'WHERE linha = \'{linha}\' '
                 f'AND EXTRACT (HOUR FROM datahora_ts) >= 8 '
                 f'AND EXTRACT (HOUR FROM datahora_ts) < 23 ')
        # query = (f'SELECT ordem, dg.latitude, dg.longitude, datahora_ts, velocidade '
        #          f'FROM dados_gps_{date} dg '
        #          f'LEFT JOIN coords_garagem cg '
        #          f'ON ST_DWithin(cg.geom, dg.geom, 200 / 111320.0) '
        #          f'WHERE cg.geom IS NULL '
        #          f'AND dg.linha = \'{linha}\' '
        #          f'AND EXTRACT (HOUR FROM dg.datahora_ts) >= 9 '
        #          f'AND EXTRACT (HOUR FROM dg.datahora_ts) <= 18 ')
                #  f'AND dg.velocidade = 0 ')
        queries.append(query)
        
    union_all_query = ' UNION ALL '.join(queries)

    df = pd.read_sql(union_all_query, con=engine)
    return df

def get_bus_stop(linha, engine):
    query = f'SELECT latitude, longitude FROM coords_final_stop WHERE linha = \'{linha}\'' 
    df = pd.read_sql(query, con=engine)
    return list(zip(df['latitude'], df['longitude']))

def create_trajectories_v3(df: pd.DataFrame, start_point: tuple, end_point: tuple) -> list[pd.DataFrame]:

    all_trajectories = []
    df.sort_values(by=['ordem', 'datahora_ts'], inplace=True)
    df = df.reset_index()

    start_point_np = np.array([start_point])
    end_point_np = np.array([end_point])
    coords = df[['latitude', 'longitude']].to_numpy()

    # Calculate the distances using the haversine formula
    def haversine_distances(coords1, coords2):
        # Convert decimal degrees to radians
        coords1 = np.radians(coords1)
        coords2 = np.radians(coords2)
        
        # Haversine formula
        dlat = coords2[:, 0] - coords1[:, 0]
        dlon = coords2[:, 1] - coords1[:, 1]
        
        a = np.sin(dlat / 2.0) ** 2 + np.cos(coords1[:, 0]) * np.cos(coords2[:, 0]) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        meters = 6371000 * c  # Radius of Earth in meters
        return meters
    
    def haversine_distance_solo(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        meters = 6371000 * c  # Radius of Earth in meters
        return meters
    
    # Calculate distances from start and end points
    distance_from_start = haversine_distances(coords, start_point_np)
    distance_from_end = haversine_distances(coords, end_point_np)
    df['distance_from_start'] = distance_from_start
    df['distance_from_end'] = distance_from_end
    
    df['datahora_ts'] = pd.to_datetime(df['datahora_ts'])

    df['prev_latitude'] = df['latitude'].shift(1)
    df['prev_longitude'] = df['longitude'].shift(1)
    df['distance_from_prev'] = haversine_distance_solo(df['latitude'], df['longitude'], df['prev_latitude'], df['prev_longitude'])
    df['distance_from_prev'] = df['distance_from_prev'].fillna(0)
    df = df.drop(columns=['prev_latitude', 'prev_longitude'])
        

    df['cumulative_distance'] = 0.0

    for ordem in df['ordem'].unique():
        df_sliced = df[df['ordem']==ordem].reset_index()
        curr_position = True
        while(True):
            start_index = -1
            for i in range(curr_position, len(df_sliced) - 1):
                if (df_sliced.iloc[i]['distance_from_start'] <= 20) and (df_sliced.iloc[i+1]['distance_from_start'] > 20):
                    start_index = i
                    break
            if start_index < 0:
                break # Não encontrou ponto inicial

            start_time = df_sliced.iloc[start_index]['datahora_ts']
            end_index = -1
            cumulative_distance = 0.0
            for i in range(start_index+1, len(df_sliced)):
                # if (df_sliced.iloc[i]['distance_from_prev'] > 2000):
                #     break
                cumulative_distance += df_sliced.iloc[i]['distance_from_prev']
                df_sliced.at[i, 'cumulative_distance'] = cumulative_distance
                if (df_sliced.iloc[i]['distance_from_end'] <= 20):
                    end_index = i
                    break 
            
            if end_index < 0: break # Não encontrou ponto final
            curr_position = end_index + 1

            final_time = pd.to_datetime(df_sliced.iloc[end_index]['datahora_ts'])
            start_time = pd.to_datetime(df_sliced.iloc[start_index]['datahora_ts'])
            if (final_time - start_time) > timedelta(hours=4): continue # Trajeto não válido
            
            trajectory_df = df_sliced.iloc[start_index:end_index+1]
            points = [(row['longitude'], row['latitude']) for index, row in trajectory_df.iterrows()]
            line = LineString(points)
            # trajectory_df['geometry'] = line
            all_trajectories.append(line)

    # if len(all_trajectories) > 0:
    #     gdf = gpd.GeoDataFrame(pd.concat(all_trajectories, ignore_index=True), crs='EPSG:4326')
    #     return gdf
    # else:
    #     return gpd.GeoDataFrame()
    return all_trajectories

def plot_list_of_linestrings(linestrings, choosen_line) -> gpd.GeoDataFrame:
    geod = Geod(ellps="WGS84")
    gdf = gpd.GeoDataFrame(geometry=linestrings, crs='EPSG:4326')
    
    gdf['route_lengths'] = gdf['geometry'].map(lambda x: geod.geometry_length(x))
    # Initialize a Folium map centered on the average location of your LineStrings
    map_center = [sum(gdf.total_bounds[[1, 3]])/2, sum(gdf.total_bounds[[0, 2]])/2]
    mymap = folium.Map(location=map_center, zoom_start=12)

    # Add LineStrings to the map
    # for idx, row in gdf.iterrows():
        
        # folium.GeoJson(row['geometry'], style_function=lambda x: {'color': 'blue'}).add_to(mymap)
  
    median_length = np.median(gdf['route_lengths'])
    
    # Find the closest LineStrings to the median length
    closest_lines = gdf.iloc[(gdf['route_lengths'] - median_length).abs().argsort()[:1]]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Add LineStrings to the map
    for idx, (color, row) in enumerate(zip(colors, closest_lines.iterrows())):
        folium.GeoJson(row[1]['geometry'], style_function=lambda x, color=color: {'color': color}).add_to(mymap)

    # mymap.save(f'maps/trajectory/linha_{choosen_line}_linestringversion.html')
    return closest_lines.drop(columns=['route_lengths'])


def upload_trajectory_to_db(gdf: gpd.GeoDataFrame, choosen_line, start_stop, final_stop, conjunto_dia: str):
    gdf['linha'] = choosen_line
    gdf['conjunto_dia'] = conjunto_dia
    gdf['ponto_inicial'] = Point(start_stop[1], start_stop[0])
    gdf['ponto_final'] = Point(final_stop[1], final_stop[0])
    gdf.rename(columns={'geometry':'trajeto'}, inplace=True)
    gdf.set_geometry("trajeto", inplace=True)
    try:
        gdf.to_postgis('rotas', db_engine_alchemy, if_exists='append', index=False)
    except Exception as e:
        if not (type(e) is psycopg2.errors.UniqueViolation):
            raise e
        else:
            print("Erro: Trajeto já inserido anteriormente no banco de dados")

def process_bus_lines_chunk(bus_lines_chunk, db_engine_alchemy):
    for bus_line in bus_lines_chunk:
        # conjunto_dias = {'WEEKDAY': SEGUNDA_FEIRA + TERCA_FEIRA + QUINTA_FEIRA}
        conjunto_dias = {'SABADO': SABADO,
                         'DOMINGO': DOMINGO}
        bus_stops = get_bus_stop(bus_line, db_engine_alchemy)
        for conjunto, dias in conjunto_dias.items():
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]Criando rota para {conjunto} da LINHA {bus_line}...")
                df = database_query(bus_line, dias, db_engine_alchemy)

                # Criando trajetórias de IDA
                trajectory_ida = create_trajectories_v3(df, bus_stops[0], bus_stops[1])
                best_route = plot_list_of_linestrings(trajectory_ida, bus_line)
                upload_trajectory_to_db(best_route, bus_line, bus_stops[0], bus_stops[1], conjunto)

                # Criando trajetórias de VOLTA
                trajectory_volta = create_trajectories_v3(df, bus_stops[1], bus_stops[0])
                best_route = plot_list_of_linestrings(trajectory_volta, bus_line)
                upload_trajectory_to_db(best_route, bus_line, bus_stops[1], bus_stops[0], conjunto)
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]ERRO na criação das rotas para {conjunto} da LINHA {bus_line}... {str(e)}")

def main():
    num_threads = 12  # Defina o número de threads desejado
    chunk_size = len(USED_BUS_LINES) // num_threads
    
    # Dividir USED_BUS_LINES em chunks
    bus_lines_chunks = [USED_BUS_LINES[i:i + chunk_size] for i in range(0, len(USED_BUS_LINES), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_bus_lines_chunk, chunk, db_engine_alchemy): chunk for chunk in bus_lines_chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"ERRO no chunk {chunk}: {e}")

if __name__ == "__main__":
    main()