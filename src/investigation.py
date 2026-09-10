# -*- coding: utf-8 -*-
'''
Analysis of seasonal phenology and climate trends for Pieris butterflies in Sweden.

This module loads, cleans, and analyzes citizen science observation data from
Artportalen (2000–2025) for two closely related butterfly species:
the green-veined white (Pieris napi) and the small white (Pieris rapae).

The analysis includes:
1. Seasonal phenology and generation peaks (bivoltinism vs. univoltinism).
2. Latitudinal effects on flight periods across different Swedish regions.
3. Long-term climate change trends by tracking the 10th percentile of spring
   emergence over a 25-year period using linear regression.

Saves three publication-quality visualization plots as PNG files.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

DATA_FOLDER = "../data/"
FIGURE_FOLDER = "../figures/"

def load_and_clean(file_path, species_name):
    ''' Load, filter, and clean butterfly observation records from a TSV file. '''
    # Läs in filen (Artportalens exporter är tab-separerade)
    df = pd.read_csv(file_path, sep='\t', low_memory=False)

    # Rensa bort observationer som saknar nödvändiga datum eller koordinater
    df = df.dropna(subset=['Startdatum', 'WGS84 decimal (lat)'])

    # Konvertera datum och skapa tidsvariabler
    df['Startdatum'] = pd.to_datetime(df['Startdatum'])
    df['Year'] = df['Startdatum'].dt.year
    df['DOY'] = df['Startdatum'].dt.dayofyear  # Day of Year (1 - 365)
    df['Latitude'] = df['WGS84 decimal (lat)']

    # Sortera bort explicita tidiga livsstadier (larver, ägg, puppor)
    # för att säkerställa att vi analyserar flygtider för vuxna individer
    stages_to_exclude = ['larv', 'ägg', 'puppa', 'larva', 'egg', 'pupa']
    df = df[~df['Stadium/ålder'].astype(str).str.lower().isin(stages_to_exclude)]

    df['Species'] = species_name
    return df[['Species', 'Year', 'DOY', 'Latitude', 'Startdatum']]

def get_zone(lat):
    ''' Vi delar in Sverige i tre geografiska zoner baserat på latitud '''
    if lat < 58.5:
        return 'Southern Sweden (< 58.5°N)'
    if lat <= 61.5:
        return 'Mid Sweden (58.5°N - 61.5°N)'
    return 'Northern Sweden (> 61.5°N)'

def main():
    ''' Main program '''
    # Sätt snygg stil för figurerna
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

    # Ladda in dataset
    napi = load_and_clean(DATA_FOLDER + 'pieris_napi.csv', 'Pieris napi (Rapsfjäril)')
    rapae = load_and_clean(DATA_FOLDER + 'pieris_rapae.csv', 'Pieris rapae (Rovfjäril)')

    # Slå ihop för gemensamma analyser
    combined = pd.concat([napi, rapae], ignore_index=True)

    # Motsvarande dagsnummer (Day of Year) för den 15:e i varje månad (april till september)
    ticks = [105, 135, 166, 196, 227, 258]
    labels = ['15 Apr', '15 Maj', '15 Jun', '15 Jul', '15 Aug', '15 Sep']

    figure_1(combined, ticks, labels)
    figure_2(combined, ticks, labels)
    figure_3(combined)
    figure_4(combined, ticks, labels)

def figure_1(combined, ticks, labels):
    ''' FIGUR 1: SÄSONGSFENOLOGI (NÄR FLYGER DE?) '''
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=combined, x='DOY', hue='Species',
                fill=True, common_norm=False,
                alpha=0.4, palette='Set1', linewidth=2)
    plt.title("Seasonal Phenology of Pieris Species in Sweden (2000–2025)")
    plt.xlabel("Day of the Year (DOY)")
    plt.ylabel("Observation Density (Proportion of Records)")
    plt.xlim(100, 280)  # Fokusera på det biologiska fönstret (ca april - oktober)

    plt.xticks(ticks, labels)

    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER + 'figure_1_seasonal_phenology.png', dpi=300)
    plt.show()

def figure_2(combined, ticks, labels):
    ''' FIGUR 2: LATITUDEFFEKT (SYD VS NORD) '''
    combined['Zone'] = combined['Latitude'].apply(get_zone)
    zone_order = ['Southern Sweden (< 58.5°N)',
                  'Mid Sweden (58.5°N - 61.5°N)',
                  'Northern Sweden (> 61.5°N)']

    g = sns.displot(
        data=combined, x='DOY', hue='Species', col='Zone', col_order=zone_order,
        kind='kde', fill=True, common_norm=False, height=5, aspect=0.9, alpha=0.3, palette='Set1'
    )
    g.set_titles("{col_name}")
    g.set_xlabels("Date")
    g.set_ylabels("Observation Density")

    # Justera x-axeln för alla delplottar
    for ax in g.axes.flat:
        ax.set_xlim(100, 280)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)

    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle("Geographical variation in flight periods across Sweden")
    g.savefig(FIGURE_FOLDER + 'figure_2_latitude_effect.png', dpi=300)
    plt.show()

def figure_3(combined):
    ''' FIGUR 3: KLIMATFÖRÄNDRINGSTREND (ÄR VÅREN TIDIGARE NU?) '''
    # Ekologiskt tips: Första vårframträdandet mäts bäst som den 10:e percentilen
    # av årets observationer för att undvika slumpmässiga extremvärden (outliers).
    trends = []
    for species in ['Pieris napi (Rapsfjäril)', 'Pieris rapae (Rovfjäril)']:
        sp_data = combined[combined['Species'] == species]
        # Gruppera per år och beräkna 10:e percentilen av DOY
        annual_first = sp_data.groupby('Year')['DOY'].quantile(0.10).reset_index()
        annual_first['Species'] = species
        trends.append(annual_first)

    trends_df = pd.concat(trends, ignore_index=True)

    plt.figure(figsize=(12, 6))
    colors = {'Pieris napi (Rapsfjäril)': '#E41A1C', 'Pieris rapae (Rovfjäril)': '#377EB8'}

    for species, color in colors.items():
        sp_trend = trends_df[trends_df['Species'] == species].dropna()

        # Scatter plot för de faktiska årliga punkterna
        plt.scatter(sp_trend['Year'], sp_trend['DOY'],
                    color=color, alpha=0.6,
                    label=f'{species} (Annual 10th %)')

        # Linjär regression
        slope, intercept, r_value, p_value, std_err = linregress(sp_trend['Year'], sp_trend['DOY'])
        years_range = np.array([sp_trend['Year'].min(), sp_trend['Year'].max()])

        # Rita trendlinjen med en mer detaljerad legend
        plt.plot(years_range, slope * years_range + intercept,
                 color=color, linestyle='--', linewidth=2,
                 label=f"{species} Trend: {slope*10:.2f} days/decade\n"
                       f"(p={p_value:.3f}, $R^2$={r_value**2:.2f}, SE={std_err:.3f})")

        # Skriv ut till terminalen för din rapportskrivning
        print("\n==================================================")
        print(f" STATISTICAL SUMMARY FOR {species.upper()}")
        print("==================================================")
        print(f"Linear Trend (slope): {slope:.3f} days/year ({slope*10:.2f} days/decade)")
        print(f"Standard Error (SE) : {std_err:.3f} days/year")
        print(f"R-squared (R2)      : {r_value**2:.3f} " + \
              f"({r_value**2*100:.1f}% of variance explained)")
        print(f"P-value             : {p_value:.5f} " + \
              ("(significant)" if p_value < 0.05 else "(not significant)"))
        print("==================================================")

    plt.title("Shift in Spring Generation Onset (10th Percentile) Over Time (2000–2025)")
    plt.xlabel("Year")
    plt.ylabel("Day of the Year for Spring Onset (DOY)")
    plt.legend(loc='upper right')
    plt.xticks(range(2000, 2026, 5))
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER + 'figure_3_climate_trend.png', dpi=300)
    plt.show()

def figure_4(combined, ticks, labels):
    ''' FIGUR 4: MILESTONE TRENDLINES (LATITUD VS FLYGTIDSMILSTOLPAR) '''
    plt.figure(figsize=(10, 6))

    colors = {
        'Pieris napi (Rapsfjäril)': '#E41A1C', 
        'Pieris rapae (Rovfjäril)': '#377EB8'
    }

    # Skapa 1-graders latitud-bins genom att runda latituden till närmaste heltal
    combined['Lat_Bin'] = combined['Latitude'].round()

    for species, color in colors.items():
        sp_data = combined[combined['Species'] == species]

        # Gruppera per latitudgrad och beräkna 10:e och 50:e percentilen
        binned = sp_data.groupby('Lat_Bin')['DOY'].agg(
            count='count',
            onset=lambda x: np.percentile(x, 10),
            peak=lambda x: np.percentile(x, 50)
        ).reset_index()

        # Filtrera bort bins med färre än 10 observationer för stabil statistik
        binned = binned[binned['count'] >= 10]

        if len(binned) > 2:
            # --- Vårstart (10:e percentilen) ---
            slope_on, inter_on, _, p_on, _ = linregress(binned['Lat_Bin'], binned['onset'])
            lat_range = np.array([binned['Lat_Bin'].min(), binned['Lat_Bin'].max()])

            # Rita faktiska beräknade punkter (cirklar) och trendlinje (streckad)
            plt.scatter(binned['Lat_Bin'], binned['onset'],
                        color=color, marker='o', alpha=0.5, s=40)
            plt.plot(lat_range, slope_on * lat_range + inter_on,
                     color=color, linestyle='--', linewidth=2,
                     label=f'{species} Start (10th %): {slope_on:.1f} days/°N (p={p_on:.3f})')

            # --- Toppnotering (50:e percentilen / Median) ---
            slope_pk, inter_pk, _, p_pk, _ = linregress(binned['Lat_Bin'], binned['peak'])

            # Rita faktiska beräknade punkter (trianglar) och trendlinje (heldragen)
            plt.scatter(binned['Lat_Bin'], binned['peak'], color=color, marker='^', alpha=0.5, s=40)
            plt.plot(lat_range, slope_pk * lat_range + inter_pk,
                     color=color, linestyle='-', linewidth=2.5,
                     label=f'{species} Peak (50th %): {slope_pk:.1f} days/°N (p={p_pk:.3f})')

            # Skriv ut statistiken i terminalen så att du enkelt kan använda den i din text
            print("\n==================================================")
            print(f" {species.upper()} - GEOGRAPHICAL TRENDS")
            print("==================================================")
            print(f"Spring Onset (10th%): {slope_on:.3f} days/°N (p = {p_on:.5f})")
            print(f"Peak Flight  (50th%): {slope_pk:.3f} days/°N (p = {p_pk:.5f})")
            print("==================================================")

    # Anpassa axlar efter Sveriges utsträckning
    plt.xlim(55, 68)
    plt.ylim(100, 280)

    # Snygga datum på y-axeln
    plt.yticks(ticks, labels)

    plt.title("Latitudinal Shifts in Butterfly Flight Milestones")
    plt.xlabel("Latitude (°N)")
    plt.ylabel("Date of Observation")
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Spara figuren i figurmappen
    plt.savefig(FIGURE_FOLDER + 'figure_4_latitude_gradient.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
