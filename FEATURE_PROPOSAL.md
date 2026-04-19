# Tri_Dashboard — Propozycja Nowych Funkcjonalności

**Projekt:** Tri_Dashboard — Comprehensive Cycling/Triathlon Performance Analysis Dashboard
**Autor:** WielkiKrzych
**Data:** Kwiecień 2026
**Repozytorium:** [github.com/WielkiKrzych/Tri_Dashboard](https://github.com/WielkiKrzych/Tri_Dashboard)

---

## Spis Treści

1. [Podsumowanie Wykonawcze](#1-podsumowanie-wykonawcze)
2. [Stan Projektu](#2-stan-projektu)
3. [Propozycje Funkcjonalności](#3-propozycje-funkcjonalności)
   - [Tier 1 — Wysoki Impakt, Niski/Medium Wysiłek](#tier-1--wysoki-impakt-istniejące-dane--fundamenty)
   - [Tier 2 — Wysoka Wartość, Umiarkowany Wysiłek](#tier-2--wysoka-wartość-umiarkowany-wysiłek)
   - [Tier 3 — Przyszłe Rozszerzenia](#tier-3--przyszłe-rozszerzenia)
4. [Mapa Drogowa Wdrożenia](#4-mapa-drogowa-wdrożenia)
5. [Analiza Konkurencyjna](#5-analiza-konkurencyjna)
6. [Podsumowanie i Następne Kroki](#6-podsumowanie-i-następne-kroki)

---

## 1. Podsumowanie Wykonawcze

Tri_Dashboard to zaawansowany dashboard do analizy wydajności kolarskiej i triathlonowej, obejmujący 28 zakładek w 4 sekcjach, zoptymalizowany pod kątem wydajności (721 testów, +1276/-808 linii w ostatnim przeglądzie). Poniższy dokument proponuje **14 nowych funkcjonalności** zorganizowanych w 3 tiry priorytetowe. Kluczowe propozycje obejmują: dedykowany wykres PMC (CTL/ATL/TSB), szacowanie VLamax z krzywej PD, longitudinalne śledzenie progów SmO2 NIRS oraz scoring gotowości treningowej HRV — wszystkie bazujące na istniejących już danych i modułach obliczeniowych. Realizacja Tier 1 pozwoli pozycjonować Tri_Dashboard jako narzędzie klasy profesjonalnej, konkurujące z TrainingPeaks i WKO5.

---

## 2. Stan Projektu

| Parametr | Wartość |
|----------|---------|
| **Stack technologiczny** | Streamlit (Python 3.10+), pandas, polars, plotly, scipy, numba, MLX, neurokit2 |
| **Skala** | ~69K LOC, 28 zakładek, 4 sekcje (Overview, Performance, Intelligence, Physiology) |
| **Architektura** | TabRegistry (OCP, lazy importlib), 58 modułów obliczeniowych, 51 modułów UI, SQLite |
| **Ostatnia optymalizacja** | Performance Review: 22 pliki zmienione, +1276/-808 linii, 721 testów przechodzących |

### Zidentyfikowane Luki

1. **Brak analizy biegu/pływania** — projekt nosi nazwę „Tri_Dashboard", ale obsługuje tylko kolarstwo
2. **Brak zakładki PMC** — `training_load.py` zawiera matematykę CTL/ATL/TSB, ale brak UI
3. **Brak importu FIT** — eksport FIT istnieje, import nie
4. **Usunięte zakładki** — Longitudinal Trends, Benchmarking, Community usunięte i nie zastąpione
5. **Moduły Genetics/Environment** istnieją, ale nie mają widocznej zakładki UI
6. **Brak dashboardu gotowości HRV** — dane HRV istnieją, ale brak porannego scoringu gotowości
7. **Brak generatora planu odżywiania** — zakładka nutrition szacuje tylko spalanie węglowodanów

### Istniejące Moduły Istotne dla Propozycji

| Moduł | Status | Uwagi |
|-------|--------|-------|
| `modules/training_load.py` | Istnieje, PMC math (ATL/CTL/TSB) | Brak dedykowanej zakładki UI |
| `modules/ui/training_load_ui.py` | Istnieje | Niepodłączony do TabRegistry |
| `modules/genetics.py` | Istnieje | Brak zakładki UI |
| `modules/ui/genetics_ui.py` | Istnieje | Niepodłączony |
| `modules/calculations/stamina.py` | Istnieje | Logika estymacji VLaMax |
| `modules/calculations/metabolic_engine.py` | Istnieje | Fenotyp + bloki treningowe |
| `modules/social/` | Istnieje | Porównanie + dane referencyjne, usunięte z zakładek |
| `modules/ui/smo2.py` | Aktywny | Wizualizacja NIRS SmO2 |
| `modules/calculations/w_prime.py` | Aktywny | Kalkulacje W' balance |
| `modules/ui/vent.py` | Aktywny | Analiza wentylacyjna z @st.fragment |

---

## 3. Propozycje Funkcjonalności

---

### Tier 1 — Wysoki Impakt, Istniejące Dane / Fundamenty

Funkcjonalności z najkorzystniejszym stosunkiem wartości do wysiłku — bazują na istniejących danych i modułach.

---

#### 1. PMC (Performance Management Chart) — Dashboard CTL/ATL/TSB

**Opis:** Dedykowana zakładka prezentująca Chronic Training Load (CTL), Acute Training Load (ATL) i Training Stress Balance (TSB) w perspektywie czasowej.

**Uzasadnienie:** Podstawa każdego narzędzia do zarządzania treningiem wytrzymałościowym — TrainingPeaks, Golden Cheetah i WKO posiadają tę funkcjonalność jako element standardowy. Matematyka PMC **już istnieje** w `training_load.py`, a UI w `training_load_ui.py` — wymaga jedynie podpięcia do TabRegistry.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Krytyczny — każdy poważny zawodnik tego oczekuje |
| **Wysiłek** | 🟢 NISKI — matematyka istnieje, wymaga UI + wykresu plotly |
| **Priorytet** | **#1 — natychmiastowa realizacja** |

**Plan implementacji:**
- **Pliki do modyfikacji:** `modules/ui/training_load_ui.py`, rejestracja w `TabRegistry`
- **Nowe elementy:** Wykres plotly (line chart) z trzema seriami: CTL (niebieski, 42-dniowa średnia), ATL (czerwony, 7-dniowa średnia), TSB (zielony/czerwony w zależności od znaku)
- **Dodatkowe funkcje:** Selektor zakresu dat, tooltipy z wartościami, marker formy (TSB > 0 = forma, TSB < -10 = zmęczenie), integracja z kalendarzem treningowym
- **Zależności:** `modules/training_load.py` (istnieje), dane TSS z sesji treningowych (istnieją)

---

#### 2. Estymacja VLamax z Krzywej Power-Duration

**Opis:** Szacowanie maksymalnej szybkości glikolitycznej (VLamax) na podstawie danych krzywej PD, z wizualizacją profilu metabolicznego.

**Uzasadnienie:** Unikalny dyferencjator — tylko WKO5 i INSCYD oferują to komercyjnie. `stamina.py` **już posiada** logikę estymacji. Pozycjonuje dashboard jako profesjonalne narzędzie treningowe.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Wysoki — wyróżnia dashboard na tle konkurencji |
| **Wysiłek** | 🟡 ŚREDNI — rozszerzenie stamina.py, nowa wizualizacja |
| **Priorytet** | **#2** |

**Plan implementacji:**
- **Pliki do modyfikacji:** `modules/calculations/stamina.py` (rozszerzenie), nowy moduł UI
- **Nowe elementy:**
  - Ekstrakcja VLamax z parametrów krzywej PD (P_max, CP, W')
  - Wykres profilu metabolicznego: udział tlenowy vs beztlenowy w produkcji mocy
  - Porównanie longitudinalne (zmiany VLamax w czasie)
  - Interpretacja: typ „puncher" (wysoki VLamax) vs „time-trialist" (niski VLamax)
- **Zależności:** `modules/calculations/stamina.py` (istnieje), dane krzywej PD (istnieją), `modules/calculations/metabolic_engine.py` (istnieje)

---

#### 3. Dekompozycja Aerobowego/Anaerobowego Wpływu Treningu

**Opis:** Rozbicie TSS każdego treningu na komponent aerobowy i anaerobowy na podstawie dystrybucji stref mocy i wydatkowania W'.

**Uzasadnienie:** Pomaga zawodnikom zrozumieć, czy trening buduje bazę tlenową, czy pojemność beztlenową — wykracza poza proste liczby TSS.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Wysoki — idzie poza standardowe TSS |
| **Wysiłek** | 🟡 ŚREDNI — nowy moduł obliczeniowy, wykorzystuje istniejące W' i strefy mocy |
| **Priorytet** | **#3** |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/training_impact.py`
- **Nowy UI:** Zakładka z wykresem słupkowym per trening + średnie kroczące
- **Algorytm:** Dystrybucja czasu w strefach → waga aerobowa/anaerobowa → TSS_aerobic + TSS_anaerobic
- **Wizualizacja:** Stacked bar chart (treningi), rolling 7-day/28-day średnia, heatmap tygodniowa
- **Zależności:** `modules/calculations/w_prime.py` (istnieje), dane stref mocy (istnieją), `modules/training_load.py` (istnieje)

---

#### 4. Scoring Gotowości Treningowej na Bazie HRV

**Opis:** Poranny scoring gotowości (readiness score) bazujący na 7-dniowej kroczącej zmienności rMSSD (CV), ze wskaźnikiem świetlnym (zielony/żółty/czerwony).

**Uzasadnienie:** Dane HRV istnieją w sekcji fizjologii, ale brak przekładalnego na działanie metryku gotowości. EnduroMetrics, Gneta i Oura oferują tę funkcjonalność jako standard.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Wysoki — bezpośrednio przekładalny na codzienne decyzje treningowe |
| **Wysiłek** | 🟢 NISKI-ŚREDNI — dane HRV istnieją, potrzebna kalkulacja rMSSD CV + prosty UI |
| **Priorytet** | **#4** |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/readiness_score.py`, moduł UI
- **Algorytm:**
  - 7-dniowa krocząca CV rMSSD (coefficient of variation)
  - Porównanie z indywidualną bazową (baseline)
  - Wskaźnik: 🟢 Gotowy (CV < 1.5 × baseline), 🟡 Ostrożnie (1.5–2.0 ×), 🔴 Odpoczynek (> 2.0 ×)
- **UI:** Traffic-light widget, tekstowa rekomendacja, wykres trendu 30-dniowego, korelacja z obciążeniem treningowym (PMC)
- **Zależności:** Dane HRV z sekcji fizjologii (istnieją), `modules/training_load.py` (dla korelacji z TSB)

---

#### 5. Longitudinalne Śledzenie Progów SmO2 NIRS

**Opis:** Śledzenie breakpointów NIRS (progi deoksygenacji SmO2) w czasie, aby obserwować ewolucję sprawności tlenowej.

**Uzasadnienie:** **Żaden konkurent** nie oferuje longitudinalnego śledzenia progów NIRS. Trainalyzed oferuje tylko analizę pojedynczej sesji. Unikalna funkcja pozycjonująca dashboard na czele innowacji.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Bardzo Wysoki — unikalna funkcja, cutting-edge |
| **Wysiłek** | 🟡 ŚREDNI — dane SmO2 istnieją, potrzeba algorytmu detekcji progów + wykresu longitudinalnego |
| **Priorytet** | **#5** |

**Plan implementacji:**
- **Pliki do modyfikacji:** `modules/ui/smo2.py` (rozszerzenie), nowy moduł obliczeniowy
- **Nowe elementy:**
  - Automatyczna detekcja breakpointów (zmiana nachylenia SmO2) per sesja — algorytm segmented regression lub Douglas-Peucker
  - Wykres longitudinalny: progi SmO2 na osi Y, data na osi X
  - Korelacja ze zmianami mocy output w tych samych punktach czasowych
  - Interpretacja: przesunięcie progu w prawo = poprawa sprawności tlenowej
- **Zależności:** `modules/ui/smo2.py` (istnieje, aktywny), dane NIRS (istnieją), scipy dla segmented regression

---

### Tier 2 — Wysoka Wartość, Umiarkowany Wysiłek

Funkcjonalności o znaczącej wartości, wymagające umiarkowanej ilości nowej logiki.

---

#### 6. MPA (Maximum Power Available) / Rozszerzenie W'bal w Czasie Rzeczywistym

**Opis:** Rozszerzenie istniejącego W' balance o koncepcję MPA — maksymalna moc chwilowa w każdym momencie jazdy na podstawie pozostałego W'.

**Uzasadnienie:** Spopularyzowane przez Sufferfest/Wahoo — pokazuje „ile paliwa jest w baku" w dowolnym momencie jazdy. Idealne do analizy treningu interwałowego.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Średnio-Wysoki — świetne do analizy interwałów |
| **Wysiłek** | 🟡 ŚREDNI — rozszerzenie w_prime.py, overlay na istniejących wykresach mocy |

**Plan implementacji:**
- **Pliki do modyfikacji:** `modules/calculations/w_prime.py` (rozszerzenie)
- **Formuła:** MPA = W'bal_remaining × CP / W' + CP
- **Nowe elementy:**
  - Krzywa MPA nałożona na wykres mocy (szary obszar powyżej mocy rzeczywistej)
  - Moment „ Exhaustion Event" — kiedy moc rzeczywista dotyka krzywej MPA
  - Statystyki: czas spędzony powyżej 90% MPA, liczba Exhaustion Events
- **Zależności:** `modules/calculations/w_prime.py` (istnieje, aktywny), dane mocy (istnieją)

---

#### 7. Wielosportowe Progi (Bieg/Pływanie)

**Opis:** Oddzielne CP/W'/strefy per dyscyplina — strefy tempa biegowego, strefy tempa pływania obok stref mocy kolarskich.

**Uzasadnienie:** Projekt nosi nazwę **„Tri_Dashboard"** — obecnie obsługuje tylko kolarstwo. Triathloniści potrzebują wszystkich trzech sportów. To fundamentalna luka.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Bardzo Wysoki — odblokowuje rzeczywisty use case triathlonu |
| **Wysiłek** | 🔴 WYSOKI — nowe pipeline'y danych dla biegu/pływania, nowe moduły obliczeniowe, nowe zakładki UI |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/run_thresholds.py`, `modules/calculations/swim_thresholds.py`, moduły UI
- **Bieg:** Strefy tempa z danych GPS (pace zones na bazie Critical Pace), analiza ułożenia terenu
- **Pływanie:** Strefy tempa z danych basenu/open water (Critical Swim Speed), analiza SWOLF
- **UI:** Ujednolicony interfejs zarządzania progami per sport, widok porównawczy 3 dyscyplin
- **Ograniczenia:** Wymaga nowych źródeł danych — import plików FIT z danymi biegowymi/pływackimi
- **Zależności:** Architektura TabRegistry, istniejący framework obliczeń kolarskich jako wzorzec

---

#### 8. Silnik Planowania Odżywiania/Nawodnienia

**Opis:** Generator planu odżywiania przed startem na podstawie estymowanego tempa spalania węglowodanów, tempa pocenia i czasu trwania wydarzenia.

**Uzasadnienie:** #1 niezaspokojona potrzeba według ankiet trenerskich. hDrop (2025) waliduje sensory potu z 92% zgodnością. Konkretna, praktyczna wartość na dzień zawodów.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Wysoki — praktyczny, zawodnicy użyją tego na dzień startu |
| **Wysiłek** | 🟡 ŚREDNI — estymacja spalania węgli istnieje, dodać model tempa pocenia + generator planu |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/fueling_engine.py`, moduł UI
- **Algorytm:**
  - Czas trwania + intensywność → godzinne zapotrzebowanie na węglowodany (g/h)
  - Estymacja tempa pocenia (ml/h) na bazie masy ciała, temperatury, intensywności
  - Rekomendacja: rodzaj żelu/napoju, częstotliwość przyjmowania, ilość wody
- **UI:** Formularz konfiguracji wydarzenia (dystans, planowane tempo, warunki), drukowalny harmonogram odżywiania z timetable
- **Zależności:** Istniejąca estymacja spalania węgli (nutrition tab), dane masy ciała z profilu

---

#### 9. Longitudinalne Śledzenie DFA alpha1 („Żywa Krzywa Mleczanowa")

**Opis:** Śledzenie progów DFA alpha1 (fraktalna korelacja zmienności HR) w czasie jako proxy dla ewolucji progu mleczanowego.

**Uzasadnienie:** Nieinwazyjna estymacja LT — „największy trend w analizie sportów wytrzymałościowych 2025-2026". Potencjalnie eliminuje potrzebę drogich testów laboratoryjnych w czasie.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🔴 Bardzo Wysoki — zastępuje drogie testy lab w dłuższej perspektywie |
| **Wysiłek** | 🟠 ŚREDNI-WYSOKI — potrzebuje modułu kalkulacji DFA alpha1 (obliczeniowo intensywny), neurokit2 może pomóc |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/dfa_alpha1.py`, moduł UI
- **Algorytm:**
  - Kalkulacja DFA alpha1 per sesja (okna 2-minutowe z overlap)
  - Identyfikacja progu: alpha1 = 0.75 (zgodnie z literaturą)
  - Mapowanie progu na odpowiadającą moc/HR
- **Wizualizacja:** Longitudinalny wykres progu DFA alpha1 → moc, korelacja z progami SmO2 i wentylacyjnymi
- **Optymalizacja:** Obliczenia mogą być wolne — rozważyć numba/MLX akcelerację, cachowanie wyników w SQLite
- **Zależności:** neurokit2 (już w zależnościach), dane HR z wysoką częstotliwością (istnieją)

---

#### 10. Trendy Efektywności Tlenowej (Stosunek Moc/HR w Czasie)

**Opis:** Śledzenie ewolucji stosunku moc-do-HR jako miary poprawy efektywności tlenowej.

**Uzasadnienie:** Proste ale potężne — malejąca moc przy tym samym HR = poprawa efektywności. EnduroMetrics nazywa to „dEF" (dynamic Efficiency Factor).

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Średni — dobre uzupełnienie PMC |
| **Wysiłek** | 🟢 NISKI — dane istnieją, prosta kalkulacja + wykres trendu |

**Plan implementacji:**
- **Nowe pliki:** `modules/calculations/aerobic_efficiency.py`, moduł UI (lub rozszerzenie istniejącej zakładki)
- **Algorytm:** Moc/HR w kluczowych strefach HR per sesja → trend w czasie (rolling average)
- **Wizualizacja:** Line chart trendu, heatmap per strefa HR, korelacja z CTL
- **Zależności:** Dane mocy + HR (istnieją), `modules/training_load.py` (dla CTL)

---

### Tier 3 — Przyszłe Rozszerzenia

Funkcjonalności długoterminowe, wymagające znaczących inwestycji lub nowych źródeł danych.

---

#### 11. Banister Performance Prediction Model

**Opis:** Wykorzystanie danych PMC (CTL/ATL) do predykcji pików formy za pomocą modelu impuls-odpowiedź Banistera.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Średni |
| **Wysiłek** | 🟡 ŚREDNI |

**Plan implementacji:**
- Model: P(t) = k₁ × CTL(t) − k₂ × ATL(t) + P₀
- Kalibracja parametrów k₁, k₂ na historycznych danych (optymalizacja scipy)
- UI: Wykres predykcji z confidence interval, „peak performance window" marker
- **Zależności:** PMC Dashboard (#1), historyczne dane wyników/testów wydolnościowych

---

#### 12. Analiza Dynamiki Biegowej

**Opis:** Czas kontaktu z podłożem, oscylacja pionowa, sztywność sprężyny nogi (leg spring stiffness) z GPS/akcelerometru.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Wysoki (dla biegaczy) |
| **Wysiłek** | 🔴 WYSOKI — wymaga nowego pipeline'u danych z akcelerometru |

**Plan implementacji:**
- Wymaga: Import Running Dynamics z Garmin (pliki FIT z accelerometer data)
- Nowe moduły: `modules/calculations/running_dynamics.py`
- Metryki: GCT, VO, LSS, bilateral balance (jeśli dostępne)
- **Uwaga:** Realizacja uzależniona od #7 (Multi-Sport Thresholds) — najpierw pipeline biegowy

---

#### 13. Integracja Sen/Odnowa

**Opis:** Import danych snu (Garmin/Apple Watch) i korelacja z gotowością treningową.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Średni |
| **Wysiłek** | 🟡 ŚREDNI |

**Plan implementacji:**
- Import: Garmin Connect API lub Apple Health export
- Metryki: Czas snu, fazy (deep/light/REM), HRV nocturnal, SpO2
- Korelacja: Sen vs gotowość HRV, sen vs zmiana CTL, sen vs jakość treningu
- **Zależności:** Scoring HRV (#4), wymaga zewnętrznego źródła danych

---

#### 14. Planer Strengthening / Periodization Planner

**Opis:** Wizualne narzędzie periodyzacji z blokami build/peak/recover.

| Parametr | Ocena |
|----------|-------|
| **Impakt** | 🟡 Średni |
| **Wysiłek** | 🔴 WYSOKI — pełne narzędzie planowania treningu |

**Plan implementacji:**
- UI: Interaktywny kalendarz z drag-and-drop bloków treningowych
- Szablony: Linear, non-linear, undulating periodization
- Integracja: Automatyczne obliczenie obciążenia per blok na bazie PMC
- **Zależności:** PMC Dashboard (#1), predykcja Banister (#11)

---

## 4. Mapa Drogowa Wdrożenia

### Faza 1 — Szybkie Zwycięstwa (2-3 tygodnie)

| # | Funkcjonalność | Wysiłek | Pliki |
|---|----------------|---------|-------|
| 1 | **PMC Dashboard** | 🟢 Niski | `training_load_ui.py` + TabRegistry |
| 10 | **Efektywność Tlenowa** | 🟢 Niski | Nowy `aerobic_efficiency.py` + UI |
| 4 | **HRV Readiness Score** | 🟢 Niski-Średni | Nowy `readiness_score.py` + UI |

**Wartość Fazy 1:** Dashboard zyskuje 3 krytyczne funkcjonalności z minimalnym wysiłkiem, bazując w pełni na istniejących danych.

### Faza 2 — Zaawansowana Analiza (4-6 tygodni)

| # | Funkcjonalność | Wysiłek | Pliki |
|---|----------------|---------|-------|
| 2 | **VLamax Estimation** | 🟡 ŚREDNI | Rozszerzenie `stamina.py` + nowy UI |
| 3 | **Dekompozycja Treningu** | 🟡 ŚREDNI | Nowy `training_impact.py` + UI |
| 6 | **MPA/W'bal Enhancement** | 🟡 ŚREDNI | Rozszerzenie `w_prime.py` |

**Wartość Fazy 2:** Dashboard oferuje analizę metaboliczną i dekompozycję treningu dostępną tylko w narzędziach premium.

### Faza 3 — Innowacje NIRS + DFA (4-8 tygodni)

| # | Funkcjonalność | Wysiłek | Pliki |
|---|----------------|---------|-------|
| 5 | **SmO2 Longitudinal Tracker** | 🟡 ŚREDNI | Rozszerzenie `smo2.py` + nowy algorytm |
| 9 | **DFA alpha1 Tracking** | 🟠 ŚREDNI-Wysoki | Nowy `dfa_alpha1.py` + UI |

**Wartość Fazy 3:** Unikalne funkcjonalności niedostępne u konkurencji — pozycjonowanie jako cutting-edge.

### Faza 4 — Rozszerzenie Ekosystemu (8-16 tygodni)

| # | Funkcjonalność | Wysiłek | Pliki |
|---|----------------|---------|-------|
| 7 | **Multi-Sport (Run/Swim)** | 🔴 WYSOKI | Nowe pipeline'y + moduły |
| 8 | **Fueling Engine** | 🟡 ŚREDNI | Nowy `fueling_engine.py` |
| 11 | **Banister Prediction** | 🟡 ŚREDNI | Nowy moduł predykcji |

**Wartość Fazy 4:** Tri_Dashboard staje się prawdziwym narzędziem triathlonowym z planowaniem odżywiania.

### Faza 5 — Funkcjonalności Przyszłościowe (16+ tygodni)

| # | Funkcjonalność | Wysiłek |
|---|----------------|---------|
| 12 | **Running Dynamics** | 🔴 WYSOKI |
| 13 | **Sleep Integration** | 🟡 ŚREDNI |
| 14 | **Periodization Planner** | 🔴 WYSOKI |

---

## 5. Analiza Konkurencyjna

| Platforma | Kluczowy Dyferencjator | Relewantność dla Tri_Dashboard |
|-----------|------------------------|---------------------------------|
| **TrainingPeaks Analyze 360** | Nakładanie danych fizjologicznych (CORE, Tymewear, hDrop) na moc/HR | Model integracji multi-sensor |
| **INSCYD x Velocity** | Automatyczne testy metaboliczne (VO2max, VLamax, MLSS z mocy) | Walidacja naszego podejścia VLamax (#2) |
| **EnduroMetrics** | dEF (dynamic Efficiency Factor), PhysioCadence, HRV readiness | Walidacja efektywności tlenowej (#10) + HRV readiness (#4) |
| **Trainalyzed** | Automatyczna detekcja progów NIRS, adaptacyjne strefy treningowe | Walidacja SmO2 threshold tracker (#5) |
| **Gneta** | AI coach czytający pełną historię Garmin, trendy HRV, sen | Przyszły kierunek AI coaching |
| **hDrop** | Sensor potu/sodu w czasie rzeczywistym (92% zgodność) | Walidacja silnika odżywiania (#8) |
| **Golden Cheetah** | Open-source, kompletny PMC, model CP | Referencyjna implementacja PMC (#1) |
| **WKO5** | VLamax estimation, individualized training zones | Bezpośredni konkurent dla #2 |

### Unikalna Propozycja Wartości Tri_Dashboard

Po realizacji Faz 1-3 Tri_Dashboard będzie jedynym **open-source'owym** narzędziem oferującym:
- ✅ PMC z dekompozycją aerobową/anaerobową
- ✅ Estymację VLamax z krzywej PD
- ✅ Longitudinalne śledzenie progów SmO2 NIRS
- ✅ DFA alpha1 tracking jako proxy krzywej mleczanowej
- ✅ Scoring gotowości HRV z rekomendacjami treningowymi

Ta kombinacja jest niedostępna w żadnym pojedynczym narzędziu — komercyjnym ani open-source.

---

## 6. Podsumowanie i Następne Kroki

### Rekomendacja Startowa

**Rozpocząć od PMC Dashboard (#1)** — najniższy wysiłek, najwyższy impakt. Matematyka istnieje w `training_load.py`, UI istnieje w `training_load_ui.py` — wymaga jedynie:
1. Rejestracji w TabRegistry
2. Wykresu plotly z trzema seriami (CTL/ATL/TSB)
3. Selektora zakresu dat

Następnie realizować Fazy 1-2 sekwencyjnie, co w ciągu ~8 tygodni dostarczy 6 wysokowartościowych funkcjonalności.

### Decyzje Wymagane

| # | Pytanie | Kontekst |
|---|---------|----------|
| 1 | **Czy realizować Multi-Sport (Faza 4)?** | Wymaga znaczącej inwestycji — czy Tri_Dashboard ma pozostać kolarski z elementami tri, czy stać się pełnym narzędziem triathlonowym? |
| 2 | **Czy dodać import FIT?** | Wymagany dla #7, #12, #13 — obecnie projekt ma tylko eksport FIT |
| 3 | **Priorytet DFA alpha1 vs Fueling Engine?** | DFA jest innowacyjny ale obliczeniowo intensywny; Fueling jest praktyczny ale mniej unikalny |
| 4 | **Czy generować zakładki Genetics/Environment?** | Moduły istnieją ale nie są podłączone — szybkie zwycięstwo jeśli zawartość jest stabilna |

### Metryki Sukcesu

- **Faza 1:** 3 nowe zakładki, 0 nowych modułów obliczeniowych (wykorzystanie istniejących)
- **Faza 2:** 3 nowe moduły obliczeniowe, dashboard na poziomie TrainingPeaks dla analizy treningu
- **Faza 3:** 2 unikalne funkcjonalności niedostępne u konkurencji
- **Docelow:** Tri_Dashboard jako jedyne open-source'owe narzędzie łączące PMC, VLamax, NIRS longitudinal i DFA alpha1

---

*Document wygenerowany w ramach analizy funkcjonalności Tri_Dashboard — Kwiecień 2026*
