select distinct partner
from wnioski
where partner is not null
;

with analiza_partnerow as (
   select w.id,
          w.typ_podrozy as typ_podrozy,
       w.jezyk,
       w.partner,
       date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 as dni_realziacji
from wnioski w, analizy_wnioskow aw
where w.id = aw.id_wniosku
and w.partner is not null
and date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) > 0
--and w.typ_podrozy is not null
)
select jezyk,
       count(id) as liczba_wnioskow,
       typ_podrozy,
       partner,
       avg(dni_realziacji) as sredni_czas_realizacji_dni,
       percentile_disc(0.5) within group (order by dni_realziacji) as mediana_dni,
       (avg(dni_realziacji)/
        percentile_disc(0.5) within group (order by dni_realziacji))::float as wspolczynnik_avg_med,
       stddev(dni_realziacji) as odchylenie_standardowe
from analiza_partnerow
group by partner,jezyk, typ_podrozy
having count(id) > 10
order by jezyk, typ_podrozy, partner;

---czas obslugi wnioskow w danym jezyku
with czas_realizacji as(
    select w.id as wniosek,
       w.jezyk as jezyk,
       date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 as dni_realziacji
    from wnioski w, analizy_wnioskow aw
    where w.id = aw.id_wniosku
    AND date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 > 0
    order by jezyk)
select jezyk,
       count(wniosek),
       AVG(dni_realziacji) as sredni_czas_realizacji_dni,
       AVG(dni_realziacji)*24.0 as sredni_czas_realizacji_godziny,
       AVG(dni_realziacji) / percentile_disc(0.5) within group ( order by dni_realziacji )::float as wspolczynnik_avg_mediana
from czas_realizacji
group by jezyk
order by sredni_czas_realizacji_dni DESC ;

------czas obslugi wnioskow w danym jezyku oraz w zaleznosci od typu podrozy (wartosc srednia)

select distinct typ_podrozy
from wnioski;

with czas_realizacji_typ_podrozy as(
    select w.id as wniosek,
       w.jezyk as jezyk,
       w.typ_podrozy as typ_podrozy,
       date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 as dni_realziacji
    from wnioski w, analizy_wnioskow aw
    where w.id = aw.id_wniosku
    AND date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 > 0
    AND w.typ_podrozy is not null
    order by jezyk)
select jezyk,
       typ_podrozy,
       count(wniosek),
       AVG(dni_realziacji) as sredni_czas_realizacji_dni,
       AVG(dni_realziacji) / percentile_disc(0.5) within group ( order by dni_realziacji )::float as wspolczynnik_avg_mediana,
       AVG(dni_realziacji)*24.0 as sredni_czas_realizacji_godziny,
       (AVG(dni_realziacji)-lag(AVG(dni_realziacji)) over (partition by jezyk))*100.0
           / lag(AVG(dni_realziacji)) over (partition by jezyk) as Zmiana_1
from czas_realizacji_typ_podrozy
group by jezyk, typ_podrozy
having count(wniosek) > 10
order by jezyk, Zmiana_1 ,typ_podrozy;

---- to co wyzej ale z nulami

with czas_realizacji_typ_podrozy as(
    select w.id as wniosek,
       w.jezyk as jezyk,
       w.typ_podrozy as typ_podrozy,
       date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 as dni_realziacji
    from wnioski w, analizy_wnioskow aw
    where w.id = aw.id_wniosku
    AND date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 > 0
    order by jezyk)
select jezyk,
       typ_podrozy,
       count(wniosek),
       AVG(dni_realziacji) as sredni_czas_realizacji_dni,
       AVG(dni_realziacji)*24.0 as sredni_czas_realizacji_godziny,
       AVG(dni_realziacji) / percentile_disc(0.5) within group ( order by dni_realziacji )::float as wspolczynnik_avg_mediana,
        (AVG(dni_realziacji)-lag(AVG(dni_realziacji)) over (partition by jezyk))*100.0
           / lag(AVG(dni_realziacji)) over (partition by jezyk) as Zmiana_1
from czas_realizacji_typ_podrozy
group by jezyk, typ_podrozy
having count(wniosek) > 10
order by jezyk, Zmiana_1 ,typ_podrozy ;


----dane do wykresu pudelkowy

with wykres_pudelkowy as(
    select w.id as wniosek,
       w.jezyk as jezyk,
       w.typ_podrozy as typ_podrozy,
       date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 as dni_realziacji
    from wnioski w, analizy_wnioskow aw
    where w.id = aw.id_wniosku
    AND date_part('hour', aw.data_zakonczenia - aw.data_utworzenia) / 24.0 > 0
    AND w.typ_podrozy is not null
    order by jezyk)
select jezyk,
       typ_podrozy,
       count(wniosek),
       percentile_disc(0.5) within group (order by dni_realziacji) as mediana_dni,
       percentile_disc(0.25) within group (order by dni_realziacji) as kwartyl_q1,
       percentile_disc(0.75) within group (order by dni_realziacji) as kwartyl_q3,
       max(dni_realziacji) as max_dni,
       min(dni_realziacji)  as min_dni
from wykres_pudelkowy
group by jezyk, typ_podrozy
order by jezyk, typ_podrozy DESC;

---analiza prawna vs jezyk wniosku i jego czas realizacji

with wnioski_vs_analiza_prawna as (
SELECT w.id as wnioski_w_sadzie,
       w.jezyk as jezyk_wnioskow,
       ap.status_sad as status_wniosku_sad,
       ap.data_rozpoczecia as data_rozp,
       ap.data_wyslania_sad as data_wysl,
       ap.data_odp_sad as data_odp
FROM wnioski w
left join analiza_prawna ap on w.id = ap.id_wniosku
WHERE ap.status_sad is not null
ORDER BY w.jezyk
    ),
     roznica_rozpoczecie_odp as (
         SELECT jezyk_wnioskow as jezyk,
                CASE WHEN extract(day from (data_odp - data_rozp))::numeric <0
                    then extract(day from (data_odp - data_rozp))::numeric*-1
                    ELSE extract(day from (data_odp - data_rozp))::numeric
                    END as czas_realizacji
         FROM wnioski_vs_analiza_prawna
     ),
     sredni_czas_realizacji_wnioskow_sad as(
         SELECT jezyk, AVG(czas_realizacji) as sredni_czas_realizacji
         FROM roznica_rozpoczecie_odp
         GROUP BY jezyk
     ),
     ilosc_wnioskow_w_danym_jezyku as (
         SELECT jezyk_wnioskow, COUNT(wnioski_w_sadzie) as ilosc_wnioskow_w_jezyku
         FROM wnioski_vs_analiza_prawna
         GROUP BY jezyk_wnioskow
     ),
     zestawienie as (
         SELECT *
         FROM ilosc_wnioskow_w_danym_jezyku iwdj, sredni_czas_realizacji_wnioskow_sad scrws
         WHERE iwdj.jezyk_wnioskow = scrws.jezyk
         ORDER BY scrws.sredni_czas_realizacji DESC
     ),
     mediana_czas_realizaccji as (
         SELECT jezyk,
               PERCENTILE_DISC(0.5) within group ( order by czas_realizacji ) as mediana_czas_realizacji
         FROM roznica_rozpoczecie_odp
         GROUP BY jezyk
     ),
     percentyle_czas_realizacji as (
         SELECT jezyk,
               PERCENTILE_DISC(0.001) within group ( order by czas_realizacji ) as Q01,
               PERCENTILE_DISC(0.05) within group ( order by czas_realizacji ) as Q5,
               PERCENTILE_DISC(0.1) within group ( order by czas_realizacji ) as Q10,
               PERCENTILE_DISC(0.2) within group ( order by czas_realizacji ) as Q20,
               PERCENTILE_DISC(0.3) within group ( order by czas_realizacji ) as Q30,
               PERCENTILE_DISC(0.5) within group ( order by czas_realizacji ) as Mediana_Q50,
               PERCENTILE_DISC(0.7) within group ( order by czas_realizacji ) as Q70,
               PERCENTILE_DISC(0.8) within group ( order by czas_realizacji ) as Q80,
               PERCENTILE_DISC(0.9) within group ( order by czas_realizacji ) as Q90,
               PERCENTILE_DISC(0.95) within group ( order by czas_realizacji ) as Q95,
               PERCENTILE_DISC(0.99) within group ( order by czas_realizacji ) as Q99
         FROM roznica_rozpoczecie_odp
         GROUP BY jezyk
     ),
     zliczanie_czasow_realizacji as (
         SELECT x.jezyk_wnioskow, COUNT(x.wnioski_w_sadzie)
         FROM wnioski_vs_analiza_prawna x
         group by jezyk_wnioskow
         HAVING COUNT(wnioski_w_sadzie) > 0
     ),
     srednia_mediana as (
     SELECT z.jezyk as jezyk,
            z.ilosc_wnioskow_w_jezyku as ilosc_wnioskow,
            z.sredni_czas_realizacji as sredni_czas_realizacji,
            mcr.mediana_czas_realizacji as mediana_czasu_realizacji,
            z.sredni_czas_realizacji / mcr.mediana_czas_realizacji::float as wspolczynnik_avg_mediana
     FROM zestawienie z, mediana_czas_realizaccji mcr
     WHERE z.jezyk_wnioskow = mcr.jezyk
         ORDER by z.sredni_czas_realizacji DESC, mcr.mediana_czas_realizacji
     )

SELECT *
FROM srednia_mediana;