     select *
from xpdusd_d;

with zapytanie_1 as (Select date_part('year', data) as rok,
       date_part('month', data) as miesiac,
       avg(zamkniecie) as srednia_miesieczne_zam
from xpdusd_d
group by rok, miesiac
order by rok, miesiac
    ),
zapytanie_2 as (
    select *,
           100*((srednia_miesieczne_zam)-lag(srednia_miesieczne_zam) over ())
               /lag(srednia_miesieczne_zam) over () as MOM
    from zapytanie_1
),
data_joining as (
    select generate_series( '2016-01-01'::date,
         date_trunc('month', '2018-11-30'::date),
         '1month'
       )::date as data_joining
),
join_data_series as(
    select dj.data_joining,
           z2.rok, z2.miesiac, z2.srednia_miesieczne_zam, z2.MOM
    from data_joining dj
    left join zapytanie_2 z2 on date_part('month', dj.data_joining)= z2.miesiac
    and date_part('year', dj.data_joining) = z2.rok
    order by z2.rok, z2.miesiac
),
data_generator as (
    select generate_series(
         '2016-01-01'::date,
         date_trunc('month', '2019-12-31'::date),
         '1month'
       )::date as wygenerowana_data
),
data_LP_num as (
    select *,
           row_number() over (order by wygenerowana_data)
    from data_generator
),
sredni_MoM as (
    select AVG(MOM)/100.0 as wspolczynnik_trendu
    from zapytanie_2
),
laczenie_tabel as (
    select dg.wygenerowana_data as data,
           coalesce(jd.srednia_miesieczne_zam, 0) as srednie_zamkniecie,
           coalesce(jd.mom, 0) as MoM
    from data_generator dg
    left join join_data_series jd on dg.wygenerowana_data = jd.data_joining
    ),
dodaj_case as (
    select *,
           CASE when MoM = 0 then 1
    else 0
    END as xyz
    from laczenie_tabel
),
dodaj_liczbe_lp as (
    select data, srednie_zamkniecie, MoM, xyz,
           case when xyz = 1 then (row_number() over ()-(select count(*) from dodaj_case
               where xyz = 0)-1) else 0 end as liczba_lp
    from dodaj_case
),
ostatnia_srednia as (
    select distinct last_value(srednie_zamkniecie) over () as last_value
    from dodaj_liczbe_lp
    where srednie_zamkniecie != 0
),
predykcja as (
    select dll.data,
           Round(case when dll.srednie_zamkniecie = 0
               then os.last_value*power((1+sr.wspolczynnik_trendu), dll.liczba_lp)
    else dll.srednie_zamkniecie
    end ,5) as predykcja_ceny
    from dodaj_liczbe_lp dll, ostatnia_srednia os, sredni_MoM sr
)
select *
from predykcja;