----- pierwsza czesc

with test_1 as (
    select sport, year, sex, season, count(*) as ilosc_NA
    from athlete_events
    where (weight = 'NA' or height = 'NA')
    --and lower(sport) like 'weightli%'
    group by year, sex, sport, season
    order by year asc, sex
),
test_2 as (
select sport, year, sex, season ,count(*) as wszystko,
       sum(case when height != 'NA' and weight != 'NA' then 1 else 0 end) as brak_NA
    from athlete_events
    --where lower(sport) like 'weightli%'
    group by year, sex, sport, season
    order by year asc, sex
),
join_test as (
select t2.sport,
       t2.year as rok_2,
       t2.sex as sex, t1.ilosc_NA as NA,
       t2.brak_NA as brak_nax,
       t2.wszystko as wszystko,
       t2.season as season,
       case when t1.ilosc_NA is null then 0 else t1.ilosc_NA end as NA_x
from test_1 t1
full join test_2 t2 on t1.sport = t2.sport
    and t1.year = t2.year
and t1.sex = t2.sex
and t1.season = t2.season
    order by t1.year asc, sport

),
obl as (
    select *,
           NA_x*100.0/wszystko as procent,
           brak_nax-NA_x::float as roznica
    from join_test
),
zapytanie_5 as (
   select *
   from obl
   where brak_nax >= 10
   and procent <50
),
zapytanie_6 as (
    select id, name, age, height, weight, sport, sex, season, year
    from athlete_events
    where weight != 'NA'

),
zapytanie_7 as (
    select sport, sex, season,  year,
           percentile_disc(0.5) within group (order by weight) as mediana_waga
    from zapytanie_6
    group by sport, year, season, sex
),
zapytanie_8 as (
    select id, name, age, height, weight, sport, sex, season, year
    from athlete_events
    where height != 'NA'
),
zapytanie_9 as (
    select sport, sex, season,  year,
           percentile_disc(0.5) within group (order by height) as mediana_wzrost
    from zapytanie_8
    group by sport, year, season, sex
),
zapytanie_10 as (
   select a.id, a.name, a.sex, a.age, a.weight, a.height, a.year, a.season, a.city, a.sport,
          zap5.sport as zap_sport, zap5.sex as zap_sex, zap5.season as zap_season, zap5.rok_2 as zap_rok
   from athlete_events a
    left join zapytanie_5 zap5 on  a.sport = zap5.sport
    where a.year = zap5.rok_2
    and a.sex = zap5.sex
    and a.season = zap5.season
    ),
zapytanie_11 as(
    select b.id, b.name, b.sex, b.age, b.weight, b.height, b.year, b.season, b.city, b.sport,
           c.mediana_waga, c.sex as sex_mediana_waga, c.sport as sport_mediana_waga, c.season as season_mediana_waga,
           c.year as year_mediana_waga
    from zapytanie_10 b
    left join zapytanie_7 c on b.sport = c.sport
    where b.year = c.year
    and b.sex = c.sex
    and b.season = c.season
    ),
zapytanie_12 as (
    select d.id, d.name, d.sex, d.age, d.weight, d.height, d.year, d.season, d.city, d.sport, d.mediana_waga,
           e.mediana_wzrost, e.sex as sex_wzrost, e.sport as sport_wzrost , e.season as season_wzrost,
           e.year as year_wzrost
    from zapytanie_11 d
    left join zapytanie_9 e on  d.sport = e.sport
    where d.year = e.year
    and d.sex = e.sex
    and d.season = e.season
    order by d.sport, d.year, d.sex
    ),
zapytanie_13 as (
    select  id, name, sex, age, weight, height, year, season, city,
           sport, mediana_waga, mediana_wzrost,
           CASE when weight = 'NA' then mediana_waga else weight
                end as waga_uzupelnione
    from zapytanie_12),
zapytanie_14 as (
    select  id, name, sex, age, weight, height, year, season, city,
           sport, mediana_waga, mediana_wzrost, waga_uzupelnione,
           CASE when height = 'NA' then mediana_wzrost else height
                end as wzrost_uzupelnione
    from zapytanie_13),
koniec as (
     select  id, name, sex, age,  year, season, city,
           sport, mediana_waga, mediana_wzrost, waga_uzupelnione, wzrost_uzupelnione
    from zapytanie_14
    ),
koniec_bmi as (
    select *,
           waga_uzupelnione::numeric/
           (((wzrost_uzupelnione::numeric)/100)*((wzrost_uzupelnione::numeric)/100)) as BMI
    from koniec
    )
select *
from koniec_bmi;

---BMI

with test_1 as (
    select sport, year, sex, season, count(*) as ilosc_NA
    from athlete_events
    where (weight = 'NA' or height = 'NA')
    --and lower(sport) like 'weightli%'
    group by year, sex, sport, season
    order by year asc, sex
),
test_2 as (
select sport, year, sex, season ,count(*) as wszystko,
       sum(case when height != 'NA' and weight != 'NA' then 1 else 0 end) as brak_NA
    from athlete_events
    --where lower(sport) like 'weightli%'
    group by year, sex, sport, season
    order by year asc, sex
),
join_test as (
select t2.sport,
       t2.year as rok_2,
       t2.sex as sex, t1.ilosc_NA as NA,
       t2.brak_NA as brak_nax,
       t2.wszystko as wszystko,
       t2.season as season,
       case when t1.ilosc_NA is null then 0 else t1.ilosc_NA end as NA_x
from test_1 t1
full join test_2 t2 on t1.sport = t2.sport
    and t1.year = t2.year
and t1.sex = t2.sex
and t1.season = t2.season
    order by t1.year asc, sport

),
obl as (
    select *,
           NA_x*100.0/wszystko as procent,
           brak_nax-NA_x::float as roznica
    from join_test
),
zapytanie_5 as (
   select *
   from obl
   where brak_nax >= 10
   and procent <50
),
zapytanie_6 as (
    select id, name, age, height, weight, sport, sex, season, year
    from athlete_events
    where weight != 'NA'

),
zapytanie_7 as (
    select sport, sex, season,  year,
           percentile_disc(0.5) within group (order by weight) as mediana_waga
    from zapytanie_6
    group by sport, year, season, sex
),
zapytanie_8 as (
    select id, name, age, height, weight, sport, sex, season, year
    from athlete_events
    where height != 'NA'
),
zapytanie_9 as (
    select sport, sex, season,  year,
           percentile_disc(0.5) within group (order by height) as mediana_wzrost
    from zapytanie_8
    group by sport, year, season, sex
),
zapytanie_10 as (
   select a.id, a.name, a.sex, a.age, a.weight, a.height, a.year, a.season, a.city, a.sport,
          zap5.sport as zap_sport, zap5.sex as zap_sex, zap5.season as zap_season, zap5.rok_2 as zap_rok
   from athlete_events a
    left join zapytanie_5 zap5 on  a.sport = zap5.sport
    where a.year = zap5.rok_2
    and a.sex = zap5.sex
    and a.season = zap5.season
    ),
zapytanie_11 as(
    select b.id, b.name, b.sex, b.age, b.weight, b.height, b.year, b.season, b.city, b.sport,
           c.mediana_waga, c.sex as sex_mediana_waga, c.sport as sport_mediana_waga, c.season as season_mediana_waga,
           c.year as year_mediana_waga
    from zapytanie_10 b
    left join zapytanie_7 c on b.sport = c.sport
    where b.year = c.year
    and b.sex = c.sex
    and b.season = c.season
    ),
zapytanie_12 as (
    select d.id, d.name, d.sex, d.age, d.weight, d.height, d.year, d.season, d.city, d.sport, d.mediana_waga,
           e.mediana_wzrost, e.sex as sex_wzrost, e.sport as sport_wzrost , e.season as season_wzrost,
           e.year as year_wzrost
    from zapytanie_11 d
    left join zapytanie_9 e on  d.sport = e.sport
    where d.year = e.year
    and d.sex = e.sex
    and d.season = e.season
    order by d.sport, d.year, d.sex
    ),
zapytanie_13 as (
    select  id, name, sex, age, weight, height, year, season, city,
           sport, mediana_waga, mediana_wzrost,
           CASE when weight = 'NA' then mediana_waga else weight
                end as waga_uzupelnione
    from zapytanie_12),
zapytanie_14 as (
    select  id, name, sex, age, weight, height, year, season, city,
           sport, mediana_waga, mediana_wzrost, waga_uzupelnione,
           CASE when height = 'NA' then mediana_wzrost else height
                end as wzrost_uzupelnione
    from zapytanie_13),
koniec as (
     select  id, name, sex, age,  year, season, city,
           sport, mediana_waga, mediana_wzrost, waga_uzupelnione, wzrost_uzupelnione
    from zapytanie_14
    ),
koniec_bmi as (
    select *,
           waga_uzupelnione::numeric/
           (((wzrost_uzupelnione::numeric)/100)*((wzrost_uzupelnione::numeric)/100)) as BMI
    from koniec
    ),
srednie_BMI as (
    select count(id) as ilosc,sex, sport, year, stddev(BMI) as sttdev_BMI
    from koniec_BMI
    group by sport, sex, year
    order by sport, year asc, sex
    )
select *
from srednie_BMI;

