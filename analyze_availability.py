# pick study people and check missing values in the first 30 features
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
CYCLES = [(2011, 'G'), (2013, 'H'), (2015, 'I'), (2017, 'J')]
MODULES = 'DEMO GHB DIQ BMX BPX PAQ SMQ ALQ DR1TOT SLQ WHQ MCQ'.split()
# diastolic values below 30 mmHg are measurement errors
MIN_DIASTOLIC_MMHG = 30.0
FEATURES = [
    ('age', 'Yaş', 'Demografi'), ('sex', 'Cinsiyet', 'Demografi'),
    ('education', 'Eğitim düzeyi', 'Demografi'), ('income_poverty_ratio', 'Gelir/yoksulluk oranı', 'Demografi'),
    ('marital_status', 'Medeni durum', 'Demografi'),
    ('height_cm', 'Boy', 'Vücut'), ('weight_kg', 'Kilo', 'Vücut'), ('bmi', 'BMI', 'Vücut'),
    ('waist_cm', 'Bel çevresi', 'Vücut'), ('waist_height_ratio', 'Bel/boy oranı', 'Vücut'),
    ('systolic_mean', 'Ortalama sistolik tansiyon', 'Tansiyon'),
    ('diastolic_mean', 'Ortalama diyastolik tansiyon', 'Tansiyon'), ('pulse', 'Nabız', 'Tansiyon'),
    ('work_activity_min_week', 'İşte haftalık aktivite süresi', 'Aktivite'),
    ('leisure_activity_min_week', 'Boş zamanda haftalık aktivite süresi', 'Aktivite'),
    ('transport_min_week', 'Ulaşımda yürüme/bisiklet süresi', 'Aktivite'),
    ('sedentary_min_day', 'Günlük oturma süresi', 'Aktivite'),
    ('smoking_status', 'Sigara kullanım durumu', 'Sigara'),
    ('cigarettes_per_smoking_day', 'İçilen günlerde günlük sigara sayısı', 'Sigara'),
    ('alcohol_frequency', 'Alkol tüketim sıklığı (ön eşleme)', 'Alkol'),
    ('drinks_per_drinking_day', 'İçilen günlerde içki sayısı', 'Alkol'),
    ('sleep_workday_hours', 'İş günlerinde uyku süresi (ön eşleme)', 'Uyku'),
    ('weight_1year_ago_kg', 'Bir yıl önce bildirilen kilo', 'Kilo geçmişi'),
    ('family_diabetes', 'Yakın ailede diyabet geçmişi', 'Aile geçmişi'),
    ('energy_kcal', 'Birinci gün enerji', 'Beslenme'), ('carbohydrate_g', 'Karbonhidrat', 'Beslenme'),
    ('protein_g', 'Protein', 'Beslenme'), ('total_fat_g', 'Toplam yağ', 'Beslenme'),
    ('total_sugar_g', 'Toplam şeker', 'Beslenme'), ('fiber_g', 'Lif', 'Beslenme'),
]


def clean_text(s):
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', s)).split())


# read html codebook and find missing / refused / don't know codes for each variable
def codebook(path):
    raw = path.read_text(encoding='utf-8-sig')
    headings = list(re.finditer(r'<h3\b[^>]*id="([^"]+)"[^>]*>(.*?)</h3>', raw, re.S | re.I))
    result = {}
    for i, match in enumerate(headings):
        code = match.group(1).upper()
        section = raw[match.end():headings[i+1].start() if i+1 < len(headings) else len(raw)]
        rows = []
        for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', section, re.S | re.I):
            cells = [clean_text(c) for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, re.S | re.I)]
            if len(cells) >= 2:
                rows.append(cells)
        missing_codes = []
        for cells in rows:
            desc = cells[1].lower().replace('’', "'")
            if any(s in desc for s in ['refused', "don't know", 'do not know', 'missing']):
                try:
                    missing_codes.append(float(cells[0]))
                except ValueError:
                    pass
        result[code] = {'label': clean_text(match.group(2)), 'missing_codes': missing_codes,
                        'rows': rows, 'text': clean_text(section)}
    return result


def main():
    OUT.mkdir(exist_ok=True)
    all_master, all_features, all_status, flow, inventory, mappings = [], [], [], [], [], []
    quality = {'errors': [], 'checks': [], 'special_codes': {}, 'xpt_zero_normalization': {}}
    for year, suffix in CYCLES:
        cycle = f'{year}-{year+1}'
        folder = ROOT / 'raw' / cycle
        data, books = {}, {}
        for module in MODULES:
            path = folder / f'{module}_{suffix}.xpt'
            frame = pd.read_sas(path, format='xport', encoding='utf-8')
            frame.columns = frame.columns.str.upper()
            # pandas reads some zeros in xpt files as 2**-260
            # change only that exact value back to 0 and count how many changed
            normalized = {}
            for col in frame.select_dtypes(include='number').columns:
                zero_artifact = frame[col].eq(np.ldexp(1.0, -260))
                if zero_artifact.any():
                    normalized[col] = int(zero_artifact.sum())
                    frame.loc[zero_artifact, col] = 0.0
            quality['xpt_zero_normalization'][f'{module}_{suffix}'] = normalized
            assert frame.SEQN.notna().all() and frame.SEQN.is_unique, f'{cycle}/{module}: SEQN hatası'
            data[module] = frame
            books[module] = codebook(folder / f'{module}_{suffix}.htm')
            assert books[module], f'{cycle}/{module}: Kod kitabı ayrıştırılamadı'
            inventory.append({'cycle': cycle, 'module': module, 'rows': len(frame), 'columns': len(frame.columns),
                              'unique_seqn': True})
            quality['special_codes'][f'{module}_{suffix}'] = {
                c: b['missing_codes'] for c, b in books[module].items() if b['missing_codes']}

        # start with DEMO table and join other modules using SEQN
        demo = data['DEMO'].set_index('SEQN')
        master = demo.copy()
        owners = {c: 'DEMO' for c in master.columns}
        for module in MODULES:
            if module == 'DEMO':
                continue
            frame = data[module].set_index('SEQN')
            assert frame.index.isin(master.index).all(), f'{cycle}/{module}: DEMO dışında kişi'
            additions = [c for c in frame.columns if c not in master.columns]
            master = master.join(frame[additions], how='left', validate='one_to_one')
            owners.update({c: module for c in additions})
        assert len(master) == len(demo)
        master.insert(0, 'cycle', cycle)

        # read column as numbers and turn refused / don't know codes into NaN
        def get(code):
            code = code.upper()
            if code not in master.columns:
                return pd.Series(np.nan, index=master.index)
            values = pd.to_numeric(master[code], errors='coerce').copy()
            module = owners[code]
            assert code in books[module], f'{cycle}: {code} için kod kitabı yok'
            return values.replace(books[module][code]['missing_codes'], np.nan)

        def track(stage, keep):
            nonlocal active
            before = int(active.sum())
            active &= keep.fillna(False)
            flow.append({'cycle': cycle, 'stage': stage, 'before': before,
                         'removed': before-int(active.sum()), 'remaining': int(active.sum())})

        # filter people step by step: age 20+, did exam, has HbA1c, no known diabetes
        active = pd.Series(True, index=master.index)
        track('00_tum_katilimcilar', active)
        track('01_yas_en_az_20', get('RIDAGEYR').ge(20))
        track('02_muayeneye_katilmis', get('RIDSTATR').eq(2))
        track('03_hba1c_mevcut', get('LBXGH').notna())
        known = get('DIQ010').isin([1, 3]) | get('DIQ160').eq(1)
        track('04_bildirilen_tani_prediyabet_yok', ~known)
        track('05_bildirilen_insulin_ilac_yok', ~(get('DIQ050').eq(1) | get('DIQ070').eq(1)))
        # people without diabetes skip the pill question (DIQ070), check codebook for this rule
        routing_verified = 'GO TO END OF SECTION' in books['DIQ']['DIQ065']['text'].upper()
        assert routing_verified, f'{cycle}: DIQ070 atlama kuralı doğrulanamadı'
        negative_diagnoses = get('DIQ010').eq(2) & get('DIQ160').eq(2)
        insulin_no = get('DIQ050').eq(2)
        pills_no_or_routed = get('DIQ070').eq(2) | (master['DIQ070'].isna() & negative_diagnoses)
        track('06_uygunluk_bilgisi_belirsiz_olanlari_ayir', negative_diagnoses & insulin_no & pills_no_or_routed)
        # exclude pregnant women (RIDEXPRG == 1 in DEMO codebook)
        track('07_hamile_degil', ~get('RIDEXPRG').eq(1))
        master['cohort_eligible'] = active
        master['reported_diagnosis'] = known
        # target is 1 when HbA1c >= 5.7, otherwise 0
        master['target'] = get('LBXGH').ge(5.7).astype('Int64').where(get('LBXGH').notna())

        features = pd.DataFrame(index=master.index)
        status = pd.DataFrame(index=master.index)

        def put(name, values, source, group_status=None):
            features[name] = values
            status[name] = pd.Series(np.where(values.notna(), 'available', 'missing'), index=master.index)
            if group_status:
                for code, mask in group_status.items():
                    status.loc[mask, name] = code
            mappings.append({'cycle': cycle, 'feature': name, 'source': source})

        # RIDAGEYR is top-coded at 80 in NHANES (80 means 80+)
        direct = {'age': 'RIDAGEYR', 'sex': 'RIAGENDR', 'education': 'DMDEDUC2',
                  'income_poverty_ratio': 'INDFMPIR', 'marital_status': 'DMDMARTL',
                  'height_cm': 'BMXHT', 'weight_kg': 'BMXWT', 'bmi': 'BMXBMI', 'waist_cm': 'BMXWAIST',
                  'pulse': 'BPXPLS', 'sedentary_min_day': 'PAD680', 'family_diabetes': 'MCQ300C'}
        for name, source in direct.items():
            values = get(source)
            if name == 'pulse':
                values = values.where(values.gt(0))
            put(name, values, source)
        put('waist_height_ratio', get('BMXWAIST') / get('BMXHT').where(get('BMXHT').gt(0)), 'BMXWAIST / BMXHT')
        # mean of first 3 blood pressure checks (do not use the 4th one)
        for name, prefix in [('systolic_mean', 'BPXSY'), ('diastolic_mean', 'BPXDI')]:
            readings = pd.concat([get(prefix+str(i)) for i in [1, 2, 3]], axis=1)
            mean_val = readings.mean(axis=1)
            if name == 'diastolic_mean':
                bad_dia = mean_val.lt(MIN_DIASTOLIC_MMHG)
                print(f"{cycle} diastolic_mean < {MIN_DIASTOLIC_MMHG}: {int((bad_dia & active).sum())} rows set to NaN", flush=True)
                mean_val = mean_val.where(~bad_dia)
            put(name, mean_val, prefix+'1,2,3; en az bir ölçüm; dördüncü dahil değil')

        # total weekly minutes = days * minutes (0 if answer is No)
        def activity(flag, days, minutes):
            answer, d, m = get(flag), get(days), get(minutes)
            value = (d*m).where(answer.eq(1) & d.between(1, 7) & m.gt(0))
            return value.mask(answer.eq(2), 0), answer.eq(2)

        work_v, work_vzero = activity('PAQ605', 'PAQ610', 'PAD615')
        work_m, work_mzero = activity('PAQ620', 'PAQ625', 'PAD630')
        lei_v, lei_vzero = activity('PAQ650', 'PAQ655', 'PAD660')
        lei_m, lei_mzero = activity('PAQ665', 'PAQ670', 'PAD675')
        transport, transport_zero = activity('PAQ635', 'PAQ640', 'PAD645')
        put('work_activity_min_week', work_v+work_m, 'PAQ605/610/PAD615 + PAQ620/625/PAD630',
            {'derived_zero': work_vzero & work_mzero})
        put('leisure_activity_min_week', lei_v+lei_m, 'PAQ650/655/PAD660 + PAQ665/670/PAD675',
            {'derived_zero': lei_vzero & lei_mzero})
        put('transport_min_week', transport, 'PAQ635/640/PAD645', {'derived_zero': transport_zero})
        # also save hard and moderate activity separately
        for name, value in [('work_vigorous_min_week', work_v), ('work_moderate_min_week', work_m),
                            ('leisure_vigorous_min_week', lei_v), ('leisure_moderate_min_week', lei_m)]:
            master[name] = value

        # smoking: 0 = never, 1 = used to smoke, 2 = smokes now
        smoke = pd.Series(np.nan, index=master.index)
        never = get('SMQ020').eq(2)
        former = get('SMQ020').eq(1) & get('SMQ040').eq(3)
        current = get('SMQ020').eq(1) & get('SMQ040').isin([1, 2])
        smoke.loc[never], smoke.loc[former], smoke.loc[current] = 0, 1, 2
        put('smoking_status', smoke, 'SMQ020 + SMQ040: 0 hiç / 1 bırakmış / 2 mevcut')
        no_smoking_days = get('SMD641').eq(0)
        smoke_na = never | former | (current & no_smoking_days)
        quantity = get('SMD650').where(current & ~no_smoking_days)
        put('cigarettes_per_smoking_day', quantity, 'SMD650; içmeyenlerde uygulanamaz', {'not_applicable': smoke_na})

        # 2017-2018 uses group codes for drinking instead of exact numbers
        # put all years into the same 0-4 groups (just a rough match)
        af = pd.Series(np.nan, index=master.index)
        structural_skip = pd.Series(False, index=master.index)
        if 'ALQ120Q' in master.columns:
            q, unit = get('ALQ120Q'), get('ALQ120U')
            annual = q * unit.map({1: 52.0, 2: 12.0, 3: 1.0})
            annual = annual.mask(q.eq(0), 0)
            af.loc[annual.eq(0)] = 0
            af.loc[annual.gt(0) & annual.lt(12)] = 1
            af.loc[annual.ge(12) & annual.lt(52)] = 2
            af.loc[annual.ge(52) & annual.lt(260)] = 3
            af.loc[annual.ge(260)] = 4
            structural_skip = get('ALQ101').eq(2) & get('ALQ110').eq(2) & q.isna()
            source = 'ALQ120Q + ALQ120U; ön ortak kategori 0–4, eğitim için kesinleştirilmedi'
        else:
            af = get('ALQ121').map({0: 0, 1: 4, 2: 4, 3: 3, 4: 3, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1, 10: 1})
            af = af.mask(get('ALQ111').eq(2), 0)
            source = 'ALQ111 + ALQ121; ön ortak kategori 0–4, eğitim için kesinleştirilmedi'
        put('alcohol_frequency', af, source, {'structural_skip': structural_skip})
        drink_na = af.eq(0)
        drinks = get('ALQ130').where(~drink_na)
        put('drinks_per_drinking_day', drinks, 'ALQ130; tüketmeyenlerde uygulanamaz; dönem üst kodları farklı',
            {'not_applicable': drink_na, 'structural_skip': structural_skip & drinks.isna()})

        # sleep column name is different in newer cycles
        sleep_code = 'SLD010H' if 'SLD010H' in master.columns else 'SLD012'
        put('sleep_workday_hours', get(sleep_code), sleep_code+'; soru/hesaplama değişikliği var, ön eşleme')
        # pounds to kg
        put('weight_1year_ago_kg', get('WHD050')/2.20462262185, 'WHD050: pound / 2.20462262185')
        # use food data only when day 1 record is reliable (DR1DRSTZ == 1)
        reliable_diet = get('DR1DRSTZ').eq(1)
        nutrients = {'energy_kcal': 'DR1TKCAL', 'carbohydrate_g': 'DR1TCARB', 'protein_g': 'DR1TPROT',
                     'total_fat_g': 'DR1TTFAT', 'total_sugar_g': 'DR1TSUGR', 'fiber_g': 'DR1TFIBE'}
        for name, source in nutrients.items():
            put(name, get(source).where(reliable_diet), source+'; DR1DRSTZ = 1 güvenilir kayıt',
                {'diet_unreliable_or_missing': ~reliable_diet})

        columns = [f[0] for f in FEATURES]
        features, status = features[columns], status[columns]
        assert list(features.columns) == columns and features.shape[1] == 30
        assert not np.isinf(features.to_numpy(dtype=float)).any()
        assert (features['age'][active] >= 20).all()
        cohort = features.loc[active].copy()
        cohort.insert(0, 'cycle', cycle)
        cohort.insert(1, 'seqn', cohort.index.astype(int))
        cohort.insert(2, 'target', master.loc[active, 'target'])
        all_features.append(cohort.reset_index(drop=True))
        selected_status = status.loc[active].copy()
        selected_status.insert(0, 'cycle', cycle)
        selected_status.insert(1, 'seqn', selected_status.index.astype(int))
        all_status.append(selected_status.reset_index(drop=True))
        master.insert(0, 'seqn', master.index.astype(int))
        all_master.append(master.reset_index(drop=True))
        quality['checks'].append({'cycle': cycle, 'cohort_size': int(active.sum()), 'unique_keys': True,
                                  'source_rows_preserved': True, 'diq070_skip_verified': routing_verified})
        print(cycle, 'toplam', len(master), 'uygun', int(active.sum()), flush=True)

    joined = pd.concat(all_master, ignore_index=True)
    cohort = pd.concat(all_features, ignore_index=True)
    statuses = pd.concat(all_status, ignore_index=True)
    assert not joined.duplicated(['cycle', 'seqn']).any()
    assert not cohort.duplicated(['cycle', 'seqn']).any()
    assert len(statuses) == len(cohort)
    assert cohort['target'].notna().all()
    assert len(joined) == sum(x['rows'] for x in inventory if x['module'] == 'DEMO')
    joined.to_csv(OUT/'merged_all_participants.csv.gz', index=False, compression='gzip')
    cohort.to_csv(OUT/'cohort_30_features.csv', index=False)
    statuses.to_csv(OUT/'feature_status.csv', index=False)
    pd.DataFrame(flow).to_csv(OUT/'cohort_flow.csv', index=False)
    pd.DataFrame(inventory).to_csv(OUT/'source_inventory.csv', index=False)
    pd.DataFrame(mappings).to_csv(OUT/'feature_mapping.csv', index=False)

    # count how many values are available or missing for each feature
    coverage = []
    for cycle in ['ALL'] + [f'{y}-{y+1}' for y, _ in CYCLES]:
        mask = pd.Series(True, index=cohort.index) if cycle == 'ALL' else cohort.cycle.eq(cycle)
        n = int(mask.sum())
        for key, label, group in FEATURES:
            s = statuses.loc[mask, key]
            numeric = int(cohort.loc[mask, key].notna().sum())
            known_na = int(s.eq('not_applicable').sum())
            coverage.append({'cycle': cycle, 'feature': key, 'label': label, 'group': group, 'n': n,
                             'numeric_or_category_available': numeric, 'available_pct': round(100*numeric/n, 2),
                             'not_applicable': known_na, 'structural_skip': int(s.eq('structural_skip').sum()),
                             'unresolved': int((~s.isin(['available', 'derived_zero', 'not_applicable'])).sum()),
                             'resolved_pct': round(100*(numeric+known_na)/n, 2)})
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT/'feature_coverage.csv', index=False)
    summary = []
    for cycle in ['ALL'] + [f'{y}-{y+1}' for y, _ in CYCLES]:
        mask = pd.Series(True, index=cohort.index) if cycle == 'ALL' else cohort.cycle.eq(cycle)
        rows = cohort.loc[mask]
        s = statuses.loc[mask, [x[0] for x in FEATURES]]
        summary.append({'cycle': cycle, 'n': len(rows), 'positive': int(rows.target.sum()),
                        'positive_pct': round(100*rows.target.mean(), 2),
                        'all30_nonmissing': int(rows[[x[0] for x in FEATURES]].notna().all(axis=1).sum()),
                        'all30_resolved_including_not_applicable': int(s.isin(['available', 'derived_zero', 'not_applicable']).all(axis=1).sum())})
    known = cohort[[x[0] for x in FEATURES]].notna()
    resolved = statuses[[x[0] for x in FEATURES]].isin(['available', 'derived_zero', 'not_applicable'])
    pd.DataFrame([{'feature_group': group,
                   'all_features_nonmissing': int(known[[k for k,l,g in FEATURES if g==group]].all(axis=1).sum()),
                   'all_applicable_resolved': int(resolved[[k for k,l,g in FEATURES if g==group]].all(axis=1).sum())}
                  for group in dict.fromkeys(x[2] for x in FEATURES)]).to_csv(OUT/'group_completeness.csv', index=False)
    quality['summary'] = summary
    quality['feature_count'] = len(FEATURES)
    quality['raw_module_count'] = len(inventory)
    quality['pregnancy_exclusion_applied'] = True
    quality['sleep_and_alcohol_harmonization'] = 'provisional; availability analysis only'
    quality['no_imputation_no_training'] = True
    (OUT/'analysis_summary.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
