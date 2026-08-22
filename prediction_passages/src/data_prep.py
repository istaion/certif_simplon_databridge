import pandas as pd
import numpy as np
from data.vacances import VACANCES_ZONE_B, JOUR_FERIE
import datetime as dt
from typing import Tuple
from trino.dbapi import connect
from trino.auth import BasicAuthentication
from dotenv import load_dotenv
import os
from src.utils_external_data import enrich_with_external_data
load_dotenv()

_ENV_TO_PREFIX: dict[str, str] = {
    "prodcentre": "wg_test_",
    "prod93":     "wg_93_",
    "prodrhone":  "wg_rhone_",
    "prod13":     "wg_13_",
}

_ENV_TO_CATALOG: dict[str, str] = {
    "prodcentre": "db_mg6jk45h_prodcentre",
    "prod93":     "db_mg6jk45h_prod93",
    "prodrhone":  "db_mg6jk45h_prodrhone",
    "prod13":     "db_mg6jk45h_prod13",
}

# Zone de vacances scolaires par défaut (fallback pour les UAI sans zone connue)
# Centre/Orléans-Tours=B, Lyon=A, Créteil(93)=C, Aix-Marseille(13)=B
_ENV_TO_FALLBACK_ZONE: dict[str, str] = {
    "prodcentre": "B",
    "prodrhone":  "A",
    "prod93":     "C",
    "prod13":     "B",
}

_ENVIRONNEMENT = os.getenv("ENVIRONNEMENT_CLIENT", "prodcentre")
_PREFIX_TABLE  = _ENV_TO_PREFIX.get(_ENVIRONNEMENT, f"wg_{_ENVIRONNEMENT}_")
_CATALOG       = _ENV_TO_CATALOG.get(_ENVIRONNEMENT, f"db_mg6jk45h_{_ENVIRONNEMENT}")


class DataPreparation:
    def __init__(self, list_site: list | None = None, exclude_holidays: bool = False, use_manual_entry: bool = True,
                 env: str | None = None, prefix: str | None = None):
        self.df = None
        self.list_site = list_site
        self.exclude_holidays = exclude_holidays
        self.use_manual_entry = use_manual_entry
        self._vacances_df = None
        self._jours_feries_df = None
        self._etablissement_detail_df = None
        self._env = env or _ENVIRONNEMENT
        self._prefix = prefix or _ENV_TO_PREFIX.get(self._env, f"wg_{self._env}_")
        self._catalog = _ENV_TO_CATALOG.get(self._env, f"db_mg6jk45h_{self._env}")
        self._fallback_zone = _ENV_TO_FALLBACK_ZONE.get(self._env, 'B')

    def load_and_prepare(self) -> pd.DataFrame:
        print(f"Récupération via le data lake (env={self._env}, prefix={self._prefix})")
        conn = connect(
            host="@data-ianord-query.eu.dataplatform.ovh.net",
            port=443,
            user=os.getenv("OVH_API_KEY"),
            auth=BasicAuthentication(os.getenv("OVH_API_KEY"), os.getenv("OVH_SECRET_KEY")),
            catalog=self._catalog,
            schema=self._env,
            http_scheme="https"
        )

        cursor = conn.cursor()
        query = f"""SELECT
                        efdate,
                        origine,
                        TRIM(CAST(codss2 AS VARCHAR)) AS codss2,
                        login_site,
                        SUM(efreel) AS efreel
                    FROM
                        {self._prefix}effect
                    WHERE
                        TRIM(LTRIM(CAST(codss1 AS VARCHAR), '0')) = '1'
                    GROUP BY
                        efdate,
                        TRIM(CAST(codss2 AS VARCHAR)),
                        login_site,
                        origine;"""
        cursor.execute(query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        self._fetch_trino_reference_data(cursor)
        df["efdate"]=pd.to_datetime(df["efdate"],format='%Y-%m-%d')
        # Supprimer les dates aberrantes (passé trop ancien ou futur)
        today = pd.Timestamp.today().normalize()
        n_future = (df["efdate"] > today).sum()
        if n_future:
            print(f"  Suppression de {n_future} lignes avec dates futures (>{today.date()})")
            df = df[df["efdate"] <= today]
        # Établissements de démonstration
        # source: wg_test_login + wg_rhone_login WHERE nometabs LIKE '%TEST]%'
        _DEMO_UAIS = {
            # prodcentre
            '0180000X', '0181111Z', '0190000A', '0280000A',
            '0280000X', '0370000X', '0410000A', '0890000A',
            # prodrhone
            '0050000A', '0070000A', '0260000A', '0300024N',
            '0380000A', '0390000A', '0420000A', '0480000A',
            '0700000A', '0740000A', '0741122B', '0840000A',
            'T700000A',
            # prod13
            '0139994T',
        }
        mask_demo = df['login_site'].isin(_DEMO_UAIS)
        n_demo = mask_demo.sum()
        if n_demo:
            print(f"  Exclusion de {df.loc[mask_demo, 'login_site'].nunique()} établissements démo ({n_demo} lignes)")
            df = df[~mask_demo]

        # df=df[df["efreel"]!=0]
        df=df[df["efreel"]>1] # Certains établissements semblent mettre 1 en saisie...
        if not self.use_manual_entry and 'origine' in df.columns:
            before = len(df)
            mask_manual = df['origine'].isna() | (df['origine'] == '') | df['origine'].str.startswith('MANUEL', na=False)
            df = df[~mask_manual]
            print(f"  Exclusion saisies manuelles: {before - len(df)} lignes retirées ({(before - len(df))/before*100:.1f}%)")

        # Agréger à (login_site, efdate, codss2) pour éliminer les doublons origine
        before = len(df)
        df = df.groupby(['efdate', 'codss2', 'login_site'], as_index=False)['efreel'].sum()
        if len(df) < before:
            print(f"  Agrégation (login_site, efdate, codss2) : {before} → {len(df)} lignes")

        if self.list_site:
            print("TYPE LIST_SITE:", type(self.list_site), self.list_site)
            df=df[df["login_site"].isin(self.list_site)]
        df["year"] = df["efdate"].dt.year
        df["month"] = df["efdate"].dt.month
        df["day"] = df["efdate"].dt.dayofweek # j'aurais du l'appeler weekday mais flemme de tout renomer...
        # début de l'année scolaire :
        start_school_year = np.where(df["month"]<8, df["year"]-1,df["year"])
        # année scolaire ex : 2023-2024
        df["school_year"] = start_school_year.astype(str)+"-"+(start_school_year+1).astype(str)

        # Enrichissement avec données externes (IPS + type établissement)
        print("Enrichissement avec données externes (IPS, type établissement)...")
        if self._etablissement_detail_df is not None:
            df = self._enrich_from_trino(df)
        else:
            df = enrich_with_external_data(df)

        df.sort_values(["login_site","codss2","efdate"], inplace=True)
        df.set_index(["login_site","codss2","efdate"],inplace=True)
        
        df['mobile_mean_42'] = df.groupby(['login_site',"codss2"])['efreel'].transform(
            lambda x: x.rolling(window=42).mean()
        )
        df['mobile_mean_42_shifted_21'] = df.groupby(['login_site',"codss2"])['mobile_mean_42'].shift(21)
        df['mobile_mean_42_shifted_21'] = df.groupby(['login_site',"codss2"])['mobile_mean_42_shifted_21'].transform(
            lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else df['efreel'].mean())
        )
        
        df['daily_mobile_mean_8'] = df.groupby(["login_site", "codss2", "day"])['efreel'].transform(
            lambda x: x.rolling(window=8).mean()
        )
        df['daily_mobile_mean_8_shifted_4'] = df.groupby(["login_site", "codss2", "day"])['daily_mobile_mean_8'].shift(4)
        df['daily_mobile_mean_8_shifted_4'] = df.groupby(["login_site", "codss2", "day"])['daily_mobile_mean_8_shifted_4'].transform(
            lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else df['efreel'].mean())
        )

        # Variante 35 jours (réutilise mobile_mean_42 et daily_mobile_mean_8 déjà calculés)
        df['mobile_mean_42_shifted_35'] = df.groupby(['login_site', "codss2"])['mobile_mean_42'].shift(35)
        df['mobile_mean_42_shifted_35'] = df.groupby(['login_site', "codss2"])['mobile_mean_42_shifted_35'].transform(
            lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else df['efreel'].mean())
        )
        df['daily_mobile_mean_8_shifted_6'] = df.groupby(["login_site", "codss2", "day"])['daily_mobile_mean_8'].shift(6)
        df['daily_mobile_mean_8_shifted_6'] = df.groupby(["login_site", "codss2", "day"])['daily_mobile_mean_8_shifted_6'].transform(
            lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else df['efreel'].mean())
        )

        df['weekday_mean'] = df.groupby(['login_site',"codss2", 'day'])['efreel'].transform('mean')
        
        df.reset_index(inplace=True)
        df.set_index(["login_site","codss2","efdate"],inplace=True)
        
        df = self.date_features(df)

        # Supprimer les observations pendant les vacances si demandé
        if self.exclude_holidays:
            initial_count = len(df)
            df = df[df["is_holyday"] == 0]
            removed_count = initial_count - len(df)
            print(f"\nSuppression des vacances: {removed_count} observations retirées ({removed_count/initial_count*100:.1f}%)")

        df = df[df["codss2"] == '2']
        df = df.sort_values(["login_site", "efdate"]).reset_index(drop=True)
        
        # Filtrer les jours avec peu de données
        df = self._filter_sparse_days(df)

        # Supprimer les outliers
        df = self._remove_outliers(df)

        # Ajouter la colonne des jours d'ouverture
        df = self.add_opening_days_column(df)

        self.df = df
        return df
    
    def _filter_sparse_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les jours de la semaine avec peu de données."""
        counts = df.groupby(['login_site', 'day']).size().reset_index(name='count')
        mean_counts = counts.groupby('login_site')['count'].mean().reset_index(name='mean_count')
        merged_counts = pd.merge(counts, mean_counts, on='login_site')
        merged_counts['threshold'] = merged_counts['mean_count'] / 4
        days_to_keep = merged_counts[merged_counts['count'] >= merged_counts['threshold']][['login_site', 'day']]

        df['join_key'] = df['login_site'].astype(str) + '_' + df['day'].astype(str)
        days_to_keep['join_key'] = days_to_keep['login_site'].astype(str) + '_' + days_to_keep['day'].astype(str)
        df_cleaned = df[df['join_key'].isin(days_to_keep['join_key'])]
        return df_cleaned.drop(columns=['join_key'])

    def _remove_outliers(self, df: pd.DataFrame, iqr_factor_lower: float = 2.5, iqr_factor_upper: float = 2.5) -> pd.DataFrame:
        """
        Supprime les outliers par site et jour de semaine en utilisant la méthode IQR.

        Un outlier est défini comme une valeur en dehors de [Q1 - iqr_factor_lower*IQR, Q3 + iqr_factor_upper*IQR]
        où IQR = Q3 - Q1.

        Args:
            df: DataFrame avec les données
            iqr_factor_lower: Multiplicateur de l'IQR pour la borne inférieure (défaut: 1.5, plus strict pour détecter les valeurs anormalement basses)
            iqr_factor_upper: Multiplicateur de l'IQR pour la borne supérieure (défaut: 2.5, plus permissif)

        Returns:
            DataFrame sans les outliers
        """
        initial_count = len(df)
        if initial_count == 0:
            return df

        # Calculer les bornes par (site, jour de semaine)
        bounds = df.groupby(['login_site', 'day'])['efreel'].agg(['median', 'std',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ])
        bounds.columns = ['median', 'std', 'q1', 'q3']
        bounds['iqr'] = bounds['q3'] - bounds['q1']
        bounds['lower'] = bounds['q1'] - iqr_factor_lower * bounds['iqr']
        bounds['upper'] = bounds['q3'] + iqr_factor_upper * bounds['iqr']

        # Assurer que lower >= 0 (pas d'effectif négatif)
        bounds['lower'] = bounds['lower'].clip(lower=0)

        bounds = bounds.reset_index()

        # Joindre les bornes au DataFrame
        df = df.merge(bounds[['login_site', 'day', 'lower', 'upper', 'median']],
                      on=['login_site', 'day'], how='left')

        # Identifier les outliers
        outlier_mask = (df['efreel'] < df['lower']) | (df['efreel'] > df['upper'])
        n_outliers = outlier_mask.sum()

        # Afficher des stats sur les outliers
        if n_outliers > 0:
            outliers_df = df[outlier_mask].copy()
            outliers_df['ecart_pct'] = ((outliers_df['efreel'] - outliers_df['median']) / outliers_df['median'] * 100).abs()

            print(f"\nSuppression des outliers (IQR factor: lower={iqr_factor_lower}, upper={iqr_factor_upper}):")
            print(f"  Outliers détectés: {n_outliers} ({n_outliers/initial_count*100:.2f}%)")

            # Top sites avec le plus d'outliers
            outliers_by_site = outliers_df.groupby('login_site').size().sort_values(ascending=False)
            print(f"  Sites les plus affectés:")
            for site, count in outliers_by_site.head(5).items():
                pct = count / len(df[df['login_site'] == site]) * 100
                print(f"    Site {site}: {count} outliers ({pct:.1f}%)")

            # Exemple d'outliers extrêmes
            extreme_outliers = outliers_df.nlargest(3, 'ecart_pct')[['login_site', 'efdate', 'day', 'efreel', 'median', 'ecart_pct']]
            print(f"  Exemples d'outliers extrêmes:")
            for _, row in extreme_outliers.iterrows():
                print(f"    Site {row['login_site']}, {row['efdate'].date()}: {row['efreel']:.0f} (médiane={row['median']:.0f}, écart={row['ecart_pct']:.0f}%)")

        # Supprimer les outliers
        df_cleaned = df[~outlier_mask].drop(columns=['lower', 'upper', 'median'])

        print(f"  Données restantes: {len(df_cleaned)} lignes ({len(df_cleaned)/initial_count*100:.1f}%)")

        return df_cleaned
    
    def train_test_split_by_date(self, df: pd.DataFrame, test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split temporel: dernier mois pour le test."""
        max_date = df["efdate"].max()
        cutoff_date = max_date - dt.timedelta(days=test_days)
        
        train = df[df["efdate"] <= cutoff_date].copy()
        test = df[df["efdate"] > cutoff_date].copy()
        
        print(f"Train: {train['efdate'].min()} -> {train['efdate'].max()} ({len(train)} lignes)")
        print(f"Test:  {test['efdate'].min()} -> {test['efdate'].max()} ({len(test)} lignes)")
        print(f"Zeros dans train: {(train['efreel'] == 0).sum()}")
        print(f"Zeros dans test: {(test['efreel'] == 0).sum()}")
        
        return train, test
    
    def _fetch_trino_reference_data(self, cursor) -> None:
        """Récupère vacances, jours fériés et détails établissements depuis Trino."""
        print("  Récupération des vacances scolaires depuis Trino...")
        cursor.execute("SELECT zone, school_year, type_vacances, date_debut, date_fin FROM default_dataset.default_dataset.vacances")
        vac = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        vac['date_debut'] = pd.to_datetime(vac['date_debut'])
        vac['date_fin'] = pd.to_datetime(vac['date_fin'])
        self._vacances_df = vac

        print("  Récupération des jours fériés depuis Trino...")
        cursor.execute("SELECT date, nom_jour_ferie FROM default_dataset.default_dataset.jours_feries")
        jf = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        jf['date'] = pd.to_datetime(jf['date'])
        self._jours_feries_df = jf

        print("  Récupération des détails établissements depuis Trino...")
        cursor.execute("SELECT uai, school_year, ips, type_etablissement, vacances_zone FROM default_dataset.default_dataset.etablissement_detail")
        etab = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        self._etablissement_detail_df = etab

    def _enrich_from_trino(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit le DataFrame avec IPS, type établissement et zone vacances depuis Trino."""
        etab = self._etablissement_detail_df

        # DEBUG -----------------------------------------------------------
        print(f"\n  [DEBUG] etablissement_detail : {len(etab)} lignes, {etab['uai'].nunique()} uai uniques")
        print(f"  [DEBUG] df : {len(df)} lignes, {df['login_site'].nunique()} login_site uniques")
        print(f"  [DEBUG] Exemples login_site : {df['login_site'].unique()[:5].tolist()}")
        print(f"  [DEBUG] Exemples uai        : {etab['uai'].unique()[:5].tolist()}")
        print(f"  [DEBUG] Exemples school_year df   : {sorted(df['school_year'].unique())[-3:]}")
        print(f"  [DEBUG] Exemples school_year etab : {sorted(etab['school_year'].unique())[-3:]}")
        sites_df = set(df['login_site'].unique())
        sites_etab = set(etab['uai'].unique())
        manquants = sites_df - sites_etab
        print(f"  [DEBUG] Sites dans df non trouvés dans etab : {len(manquants)}/{len(sites_df)}")
        if manquants:
            print(f"  [DEBUG] Exemples manquants : {list(manquants)[:10]}")
        # FIN DEBUG -------------------------------------------------------

        df = df.merge(
            etab[['uai', 'school_year', 'ips', 'type_etablissement', 'vacances_zone']],
            left_on=['login_site', 'school_year'],
            right_on=['uai', 'school_year'],
            how='left'
        ).drop(columns=['uai'])

        # Hardcode pour les sites absents de l'annuaire et des CSV IPS
        _HARDCODED = {
            '0180000X': {'vacances_zone': 'B', 'type_etablissement': "lycée agricole"},
            '0451463W': {'vacances_zone': 'B', 'type_etablissement': "lycée agricole"},
            '0410000A': {'vacances_zone': 'B', 'type_etablissement': "lycée agricole"},
        }
        for uai, vals in _HARDCODED.items():
            mask = (df['login_site'] == uai) & df['vacances_zone'].isna()
            if mask.any():
                df.loc[mask, 'vacances_zone'] = vals['vacances_zone']
                if vals['type_etablissement'] is not None:
                    df.loc[mask & df['type_etablissement'].isna(), 'type_etablissement'] = vals['type_etablissement']

        # Fallback par login_site seul (indépendant de l'année scolaire)
        # type_etablissement et vacances_zone ne varient pas dans le temps → dernière valeur connue
        site_latest = (
            etab.sort_values('school_year')
            .groupby('uai')[['ips', 'type_etablissement', 'vacances_zone']]
            .last()
            .reset_index()
            .rename(columns={
                'uai': 'login_site',
                'ips': 'ips_fallback',
                'type_etablissement': 'type_fallback',
                'vacances_zone': 'zone_fallback',
            })
        )
        df = df.merge(site_latest, on='login_site', how='left')
        df['ips'] = df['ips'].fillna(df['ips_fallback'])
        df['type_etablissement'] = df['type_etablissement'].fillna(df['type_fallback'])
        df['vacances_zone'] = df['vacances_zone'].fillna(df['zone_fallback'])
        df = df.drop(columns=['ips_fallback', 'type_fallback', 'zone_fallback'])

        n_total = len(df)
        n_ips = df['ips'].notna().sum()
        n_type = df['type_etablissement'].notna().sum()
        n_zone = df['vacances_zone'].notna().sum()
        print(f"  IPS : {n_ips}/{n_total} renseignées ({n_ips/n_total*100:.1f}%)")
        print(f"  Type établissement : {n_type}/{n_total} renseignées")
        print(f"  Zone vacances : {n_zone}/{n_total} renseignées")

        return df

    def _compute_is_holyday_trino(self, df: pd.DataFrame) -> pd.Series:
        """Calcule is_holyday par site selon leur zone de vacances (depuis Trino).
        Fallback zone B pour les sites sans zone connue."""
        result = pd.Series(0, index=df.index)
        vacances = self._vacances_df

        for zone in df['vacances_zone'].dropna().unique():
            vac_zone = vacances[vacances['zone'] == zone]
            mask_zone = df['vacances_zone'] == zone
            dates = df.loc[mask_zone, 'efdate']
            is_hol = pd.Series(False, index=dates.index)
            for _, row in vac_zone.iterrows():
                is_hol |= (dates >= row['date_debut']) & (dates <= row['date_fin'])
            result.loc[mask_zone] = is_hol.astype(int)

        mask_no_zone = df['vacances_zone'].isna()
        if mask_no_zone.any():
            vac_fallback = vacances[vacances['zone'] == self._fallback_zone]
            dates_no_zone = df.loc[mask_no_zone, 'efdate']
            is_hol_fallback = pd.Series(False, index=dates_no_zone.index)
            for _, row in vac_fallback.iterrows():
                is_hol_fallback |= (dates_no_zone >= row['date_debut']) & (dates_no_zone <= row['date_fin'])
            result.loc[mask_no_zone] = is_hol_fallback.astype(int)

        return result

    @staticmethod
    def _compute_distance_vacances(dates: pd.Series, periods: list) -> tuple:
        """Calcule jours_avant_vacance et jours_apres_vacance (cap 30) pour une série de dates.
        periods: liste de (date_debut, date_fin) en Timestamp.
        Pendant les vacances: les deux valent 0.
        """
        CAP = 30
        dates_arr = dates.values.astype('datetime64[D]')
        avant = np.full(len(dates), CAP, dtype=float)
        apres = np.full(len(dates), CAP, dtype=float)
        in_vac = np.zeros(len(dates), dtype=bool)

        for debut, fin in periods:
            d_np = np.datetime64(debut, 'D')
            f_np = np.datetime64(fin, 'D')

            in_period = (dates_arr >= d_np) & (dates_arr <= f_np)
            in_vac |= in_period

            before = dates_arr < d_np
            if before.any():
                days = ((d_np - dates_arr[before]) / np.timedelta64(1, 'D')).astype(int)
                avant[before] = np.minimum(avant[before], days)

            after = dates_arr > f_np
            if after.any():
                days = ((dates_arr[after] - f_np) / np.timedelta64(1, 'D')).astype(int)
                apres[after] = np.minimum(apres[after], days)

        avant[in_vac] = 0
        apres[in_vac] = 0

        return (
            pd.Series(np.clip(avant, 0, CAP), index=dates.index),
            pd.Series(np.clip(apres, 0, CAP), index=dates.index),
        )

    def _compute_distance_vacances_trino(self, df: pd.DataFrame) -> tuple:
        """Calcule jours_avant_vacance et jours_apres_vacance par zone depuis Trino."""
        CAP = 30
        result_avant = pd.Series(float(CAP), index=df.index)
        result_apres = pd.Series(float(CAP), index=df.index)
        vacances = self._vacances_df

        def periods_for_zone(zone):
            vz = vacances[vacances['zone'] == zone]
            return list(zip(vz['date_debut'], vz['date_fin']))

        for zone in df['vacances_zone'].dropna().unique():
            mask = df['vacances_zone'] == zone
            av, ap = self._compute_distance_vacances(df.loc[mask, 'efdate'], periods_for_zone(zone))
            result_avant[mask] = av
            result_apres[mask] = ap

        mask_no_zone = df['vacances_zone'].isna()
        if mask_no_zone.any():
            av, ap = self._compute_distance_vacances(
                df.loc[mask_no_zone, 'efdate'], periods_for_zone(self._fallback_zone)
            )
            result_avant[mask_no_zone] = av
            result_apres[mask_no_zone] = ap

        return result_avant, result_apres

    def date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df.reset_index(inplace=True)

        if self._vacances_df is not None and 'vacances_zone' in df.columns:
            df["is_holyday"] = self._compute_is_holyday_trino(df)
            df["jours_avant_vacance"], df["jours_apres_vacance"] = self._compute_distance_vacances_trino(df)
        else:
            list_holiday = VACANCES_ZONE_B
            for item in list_holiday:
                item[0], item[1] = pd.to_datetime(item[0]), pd.to_datetime(item[1])
            df["is_holyday"] = df["efdate"].apply(lambda x: 1 if self.check_is_holiday(x, list_holiday) else 0)
            periods = [(h[0], h[1]) for h in list_holiday]
            df["jours_avant_vacance"], df["jours_apres_vacance"] = self._compute_distance_vacances(df["efdate"], periods)

        if self._jours_feries_df is not None:
            ferie_dates = pd.to_datetime(self._jours_feries_df['date']).dt.normalize()
            df["is_ferie"] = df["efdate"].isin(ferie_dates).astype(int)
            list_bridge = []
            for ferie_date in ferie_dates:
                weekday = ferie_date.weekday()
                if weekday == 1:
                    list_bridge.append(ferie_date - dt.timedelta(days=1))
                elif weekday == 3:
                    list_bridge.append(ferie_date + dt.timedelta(days=1))
            df["is_bridge"] = df["efdate"].isin(pd.to_datetime(list_bridge)).astype(int)
        else:
            list_ferie = pd.to_datetime(JOUR_FERIE)
            list_bridge = []
            for ferie_date in list_ferie:
                weekday = ferie_date.weekday()
                if weekday == 1:
                    list_bridge.append(ferie_date - dt.timedelta(days=1))
                elif weekday == 3:
                    list_bridge.append(ferie_date + dt.timedelta(days=1))
            df["is_ferie"] = df["efdate"].apply(lambda x: 1 if self.check_is_same_date(x, list_ferie) else 0)
            df["is_bridge"] = df["efdate"].apply(lambda x: 1 if self.check_is_same_date(x, list_bridge) else 0)

        return df
        
    @staticmethod
    def check_is_holiday(date, list_holiday):
        for item in list_holiday:
            if date >=item[0] and date <= item[1]:
                return True
        return False
    
    @staticmethod
    def check_is_same_date(date, list_date):
        for item in list_date:
            if date.year == item.year and (date.month == item.month and date.day == item.day):
                return True
        return False

    def detect_opening_days(self, login_site: str, school_year: str, min_threshold: float = 0.3) -> str:
        """
        Détecte les jours d'ouverture de la cantine pour un établissement et une année scolaire.

        Args:
            login_site: Code de l'établissement
            school_year: Année scolaire au format "2023-2024"
            min_threshold: Seuil minimum de fréquence pour considérer un jour comme ouvert (0.3 = 30%)

        Returns:
            str: Chaîne de 7 caractères représentant les jours d'ouverture (ex: "1101100")
                 Position 0 = Lundi, 1 = Mardi, ..., 6 = Dimanche
        """
        if self.df is None:
            raise ValueError("Les données doivent être chargées avant d'appeler cette fonction. Utilisez load_and_prepare().")

        # Filtrer les données pour l'établissement et l'année scolaire
        df_filtered = self.df[
            (self.df['login_site'] == login_site) &
            (self.df['school_year'] == school_year)
        ].copy()

        if len(df_filtered) == 0:
            raise ValueError(f"Aucune donnée trouvée pour le site {login_site} et l'année scolaire {school_year}")

        # Compter le nombre de jours pour chaque jour de la semaine (0=Lundi, 6=Dimanche)
        day_counts = df_filtered.groupby('day').size()

        # Calculer le nombre total de semaines dans la période
        date_range = (df_filtered['efdate'].max() - df_filtered['efdate'].min()).days
        num_weeks = max(date_range / 7, 1)

        # Créer la chaîne de jours d'ouverture
        opening_pattern = ""
        for day in range(7):  # 0 = Lundi, 6 = Dimanche
            if day in day_counts.index:
                # Fréquence moyenne par semaine
                frequency = day_counts[day] / num_weeks
                # Si la fréquence est supérieure au seuil, considérer le jour comme ouvert
                opening_pattern += "1" if frequency >= min_threshold else "0"
            else:
                opening_pattern += "0"

        return opening_pattern

    def add_opening_days_column(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Ajoute une colonne 'opening_days' au DataFrame avec le pattern d'ouverture pour chaque établissement/année.

        Args:
            df: DataFrame à enrichir (utilise self.df si None)

        Returns:
            pd.DataFrame: DataFrame enrichi avec la colonne 'opening_days'
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("Les données doivent être chargées avant d'appeler cette fonction.")

        # Créer un dictionnaire pour stocker les patterns par (login_site, school_year)
        opening_patterns = {}

        for (login_site, school_year), group in df.groupby(['login_site', 'school_year']):
            # Compter le nombre de jours pour chaque jour de la semaine
            day_counts = group.groupby('day').size()

            # Calculer le nombre total de semaines dans la période
            date_range = (group['efdate'].max() - group['efdate'].min()).days
            num_weeks = max(date_range / 7, 1)

            # Créer la chaîne de jours d'ouverture
            opening_pattern = ""
            for day in range(7):
                if day in day_counts.index:
                    frequency = day_counts[day] / num_weeks
                    opening_pattern += "1" if frequency >= 0.3 else "0"
                else:
                    opening_pattern += "0"

            opening_patterns[(login_site, school_year)] = opening_pattern

        # Ajouter la colonne au DataFrame
        df['opening_days'] = df.apply(
            lambda row: opening_patterns.get((row['login_site'], row['school_year']), "0000000"),
            axis=1
        )
        
        df['is_day_usually_open'] = df.apply(
            lambda row: 0 if (row['is_holyday'] == 1 or row['is_ferie'] == 1) else (int(row['opening_days'][row['day']]) if len(row['opening_days']) > row['day'] else 0),
            axis=1
        )

        # Ajouter les groupes par taille
        df = self.add_size_groups(df)

        return df

    def add_size_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute une colonne 'group_size' basée sur les quartiles d'effectif moyen par site.
        Groupes: XS (très petit), S (petit), M (moyen), L (grand)
        """
        # Calculer la moyenne par site
        site_means = df.groupby('login_site')['efreel'].mean()

        # Calculer les quantiles
        q25 = site_means.quantile(0.25)
        q50 = site_means.quantile(0.50)
        q75 = site_means.quantile(0.75)

        def assign_group(mean_eff):
            if mean_eff <= q25:
                return 'XS'
            elif mean_eff <= q50:
                return 'S'
            elif mean_eff <= q75:
                return 'M'
            else:
                return 'L'

        # Créer le mapping site -> groupe
        site_to_group = site_means.apply(assign_group).to_dict()

        # Ajouter la colonne au DataFrame
        df['group_size'] = df['login_site'].map(site_to_group)

        # Afficher les statistiques
        print(f"\nGroupes par taille (seuils: XS<={q25:.0f}, S<={q50:.0f}, M<={q75:.0f}, L>{q75:.0f}):")
        for group in ['XS', 'S', 'M', 'L']:
            sites_in_group = [s for s, g in site_to_group.items() if g == group]
            group_mean = site_means[sites_in_group].mean() if sites_in_group else 0
            print(f"  {group}: {len(sites_in_group)} sites, effectif moyen={group_mean:.0f}")

        return df
