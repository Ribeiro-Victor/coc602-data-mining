delete from "dados_gps" 
where linha not in (select distinct(linha) from dados_predict);

update dados_gps  
set datahora_ts  = to_timestamp(datahora/1000),
	datahoraenvio_ts = to_timestamp(datahoraenvio/1000),
	datahoraservidor_ts = to_timestamp(datahoraservidor/1000);

select reltuples::bigint as estimate from pg_catalog.pg_class where relname = 'dados_gps';

select ordem, latitude, longitude, datahora_ts, velocidade, linha from dados_gps
where 
	extract (hour from datahora_ts) >= 1
	and
	extract (hour from datahora_ts) <= 3;

CREATE TABLE public.dados_gps (
	ordem varchar NULL,
	latitude float4 NULL,
	longitude float4 NULL,
	datahora bigint NULL,
	velocidade int NULL,
	linha varchar NULL,
	datahoraenvio bigint NULL,
	datahoraservidor bigint null,
	datahora_ts timestamp without time zone NULL,
	datahoraenvio_ts timestamp without time zone NULL,
	datahoraservidor_ts timestamp without time zone NULL
)
PARTITION BY RANGE(datahora_ts);


CREATE TABLE public.dados_gps_treino (
	ordem varchar NULL,
	latitude float4 NULL,
	longitude float4 NULL,
	datahora bigint NULL,
	velocidade int NULL,
	linha varchar NULL,
	datahoraenvio bigint NULL,
	datahoraservidor bigint null,
	datahora_ts timestamp without time zone NULL,
	datahoraenvio_ts timestamp without time zone NULL,
	datahoraservidor_ts timestamp without time zone NULL
)
PARTITION BY RANGE(datahora_ts);

CREATE TABLE public.dados_predict (
	ordem varchar NULL,
	linha varchar NULL,
	latitude float8 NULL,
	longitude float8 NULL,
	"date" varchar NULL,
	datahora int8 NULL
);


ALTER TABLE dados_gps ADD COLUMN geom geometry(Point, 4326);

UPDATE dados_gps SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);


SELECT 
    ordem, dg.latitude, dg.longitude, datahora_ts, velocidade, dg.geom
FROM 
    dados_gps_20240505 dg
LEFT JOIN 
    coords_garagem cg
ON 
    ST_DWithin(cg.geom, dg.geom, 200 / 111320.0)
WHERE 
    cg.geom IS NULL
and dg.linha = '100'


