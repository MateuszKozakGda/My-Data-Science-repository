
---Zadanie domowe
--Autor Mateusz Kozak
with glowny_wniosek as(
    select w.id as id_wniosku,
           w.kwota_rekompensaty as kwota_rekompensaty_glowna,
           w.zrodlo_polecenia as zrodlo_glowne,
           w.powod_operatora as powod_rekompensaty_glowny,
           w.data_utworzenia as data_glowna,
           w.stan_wniosku,
           sr.id_rekompensaty as id_rekompensaty_glowny,
           sr.konto as konto
    from wnioski w, szczegoly_rekompensat sr, rekompensaty r
    where w.id = r.id_wniosku
    and sr.id_rekompensaty = r.id
    and zrodlo_polecenia is null
    and w.stan_wniosku like 'wyp%'
    ),
     podzapytanie_1 as (
         select w.id as id_wniosku_pd1,
           w.kwota_rekompensaty as kwota_rekompensaty_glowna_pd1,
           w.zrodlo_polecenia as zrodlo_glowne_pd1,
           w.powod_operatora as powod_rekompensaty_glowny_pd1,
           w.data_utworzenia as data_glowna_pd1,
           w.stan_wniosku as stan_wniosku_pd1,
           sr.id_rekompensaty as id_rekompensaty_glowny_pd1,
           sr.konto as konto_pd1
         from wnioski w, szczegoly_rekompensat sr, rekompensaty r
         where w.id = r.id_wniosku
         and sr.id_rekompensaty = r.id
         and zrodlo_polecenia is not null
         and w.stan_wniosku like 'wyp%'
     ),
polaczenie as (
select gw.id_wniosku as wniosek_glowny,
       pd1.id_wniosku_pd1 as wniosek_podobny,
       gw.konto as konto_glowne,
       pd1.konto_pd1 as konto_podobne,
       gw.zrodlo_glowne as glowne_zrodlo,
       pd1.zrodlo_glowne_pd1 as podobne_zrodlo,
       gw.kwota_rekompensaty_glowna as kwota_glowna,
       pd1.kwota_rekompensaty_glowna_pd1 as podobna_kwota,
       gw.data_glowna as data_utworzenia_glowna,
       pd1.data_glowna_pd1 as data_utworzenia_podobna,
       abs(gw.kwota_rekompensaty_glowna - pd1.kwota_rekompensaty_glowna_pd1) as roznica_kwot
from glowny_wniosek gw
    left join podzapytanie_1 pd1
    on gw.konto = pd1.konto_pd1
    and gw.powod_rekompensaty_glowny = pd1.powod_rekompensaty_glowny_pd1
    and gw.data_glowna < pd1.data_glowna_pd1
    order by gw.id_wniosku, abs(gw.kwota_rekompensaty_glowna - pd1.kwota_rekompensaty_glowna_pd1) asc
),
     zestawienie_wynikow as (
         select wniosek_glowny,
                kwota_glowna as kwota_rekompensaty,
                wniosek_podobny,
                podobne_zrodlo as zrodlo_wniosku_podobnego
         from polaczenie
     )
---Odpowiedz do punktu 1:
--select *
--from zestawienie_wynikow
--limit 1;
select count(1) as suma,
       avg(kwota_rekompensaty) as srednia_kwota_rekompensaty
from zestawienie_wynikow
group by wniosek_glowny
order by 1 desc;