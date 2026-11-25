# make the extra survey, meal, smoking, and sweet drink columns

import pandas as pd

from add_symptom_features import read_fields
from analyze_availability import ROOT

YES_NO = (1, 2)
DIFFICULTY = range(1, 6)
DIRECT = [
    ("attempted_weight_loss_1y", "Kilo vermeye çalışma", "WHQ", "WHQ070", YES_NO, "Son yıl. 1 = Evet, 2 = Hayır. Soru atlananlar boş."),
    ("self_rated_diet_quality", "Beslenme değerlendirmesi", "DBQ", "DBQ700", range(1, 6), "1 = Mükemmel, 2 = Çok iyi, 3 = İyi, 4 = Orta, 5 = Kötü."),
    ("mouth_aching_frequency_1y", "Ağız ağrısı sıklığı", "OHQ", "OHQ620", range(1, 6), "30 yaş ve üzeri. Son yıl. 1 = Çok sık, 2 = Oldukça sık, 3 = Bazen, 4 = Neredeyse hiç, 5 = Hiç."),
    ("floss_days_7d", "Diş arası temizliği (gün)", "OHQ", "OHQ870", range(8), "30 yaş ve üzeri. Son 7 günde diş ipi veya benzeri araç kullanılan gün sayısı."),
    ("sleep_problems_or_oversleep_frequency", "Uyku güçlüğü veya fazla uyuma", "DPQ", "DPQ030", range(4), "Son iki hafta. Uykuya dalma, uykuyu sürdürme veya fazla uyuma. 0 = Hiç, 1 = Birkaç gün, 2 = Günlerin yarısından fazla, 3 = Neredeyse her gün."),
    ("walking_difficulty_without_equipment", "Ekipmansız yürüme güçlüğü", "PFQ", "PFQ054", YES_NO, "Sağlık sorunu nedeniyle özel ekipman olmadan yürümekte güçlük. 1 = Evet, 2 = Hayır. Soru atlananlar boş."),
    ("adult_food_security", "Yetişkin gıda güvencesi", "FSQ", "FSDAD", range(1, 5), "Son yıl. 1 = Tam, 2 = Sınırda, 3 = Düşük, 4 = Çok düşük."),
    ("employment_status_last_week", "Geçen hafta çalışma durumu", "OCQ", "OCD150", range(1, 5), "1 = Çalıştı, 2 = İşi vardı ancak çalışmadı, 3 = İş arıyordu, 4 = Çalışmıyordu."),
    ("serious_vision_difficulty", "Ciddi görme güçlüğü", "DLQ", "DLQ020", YES_NO, "2013–2018. Görme kaybı veya gözlükle bile ciddi görme güçlüğü. 1 = Evet, 2 = Hayır."),
    ("taste_difficulty_1y", "Tat alma güçlüğü", "CSQ", "CSQ080", YES_NO, "2011–2014, 40 yaş ve üzeri. Son yıl. 1 = Evet, 2 = Hayır."),
    ("persistent_mouth_sensation_1y", "Ağızda geçmeyen tat veya his", "CSQ", "CSQ110", YES_NO, "2011–2014, 40 yaş ve üzeri. Son yıl. 1 = Evet, 2 = Hayır."),
]
MEALS = [
    ("away_from_home_meals_7d", "Ev dışında hazırlanan öğün", "DBD895"),
    ("fast_food_pizza_meals_7d", "Fast-food veya pizzacı öğünü", "DBD900"),
    ("ready_to_eat_foods_30d", "Yemeye hazır yiyecek", "DBD905"),
    ("frozen_meals_pizza_30d", "Donmuş öğün veya pizza", "DBD910"),
]


def sweetened_drink_grams(folder, suffix, people):
    food_codes = pd.read_excel(folder / "wweia_food_categories.xlsx", usecols=["food_code", "category_number"])
    categories = food_codes.set_index("food_code")["category_number"]
    assert categories.index.is_unique
    recalls = pd.read_sas(folder / f"DR1IFF_{suffix}.xpt", format="xport")
    recalls["SEQN"] = recalls["SEQN"].astype(int)
    # keep only good day 1 food records (DR1DRSTZ == 1)
    recalls = recalls.loc[recalls.SEQN.isin(people) & recalls.DR1DRSTZ.round().eq(1)].copy()
    category = recalls.DR1IFDCD.round().map(categories)
    if category.isna().any():
        raise ValueError(f"{folder.name}: USDA gıda kodunda eksik eşleme.")
    # 7202, 7204, 7206 are sugary drinks in usda table (diet drinks have other codes)
    grams = recalls.DR1IGRMS.where(category.isin([7202, 7204, 7206]), 0)
    return grams.groupby(recalls.SEQN).sum(min_count=1).reindex(people)


def cycle_features(base, year, suffix):
    folder = ROOT / "raw" / f"{year}-{year + 1}"
    data = base.set_index("seqn")
    features, guide = {}, {}

    def put(name, title, code, note, values):
        features[name] = values
        guide[name] = [title, code, note]

    def read(module, *codes):
        return read_fields(folder, f"{module}_{suffix}", list(codes)).reindex(data.index)

    for name, title, module, code, valid, note in DIRECT:
        # DLQ is only from 2013 and CSQ is only in 2011-2014
        available = module not in ("DLQ", "CSQ") or (module == "DLQ" and year >= 2013) or (module == "CSQ" and year <= 2013)
        values = read(module, code)[code].round() if available else pd.Series(index=data.index, dtype=float)
        put(name, title, code, note, values.where(values.isin(valid)))

    weight = read("WHQ", "WHD020", "WHD110", "WHD140", "WHQ060")
    # NHANES skips WHQ070 when person already said they lost 10+ lb on purpose (WHQ060 == 1)
    features["attempted_weight_loss_1y"] = features["attempted_weight_loss_1y"].mask(
        features["attempted_weight_loss_1y"].isna() & weight.WHQ060.round().eq(1), 1.0
    )
    # 10-year weight change in kg, only for people age 36+
    change = (weight.WHD020.where(weight.WHD020.gt(0)) - weight.WHD110.where(weight.WHD110.gt(0))) * 0.45359237
    put("reported_weight_change_10y_kg", "10 yıllık kilo değişimi (kg)", "WHD020 − WHD110", "36 yaş ve üzeri. Güncel bildirilen kilo eksi 10 yıl önceki bildirilen kilo. Pozitif değer kilo artışıdır.", change.where(data.age.ge(36)))
    put("highest_ever_weight_kg", "Bildirilen en yüksek kilo (kg)", "WHD140", "Bildirilen yaşam boyu en yüksek kilo, pound değerinden kg'ye çevrildi.", weight.WHD140.where(weight.WHD140.gt(0)) * 0.45359237)
    meals = read("DBQ", *(code for _, _, code in MEALS))
    for name, title, code in MEALS:
        values = meals[code].round()
        weekly = code in ("DBD895", "DBD900")
        cap = 22 if weekly else 91
        # nhanes uses 5555 for 22+/week and 6666 for 91+/month
        values = values.replace({5555: 22} if weekly else {6666: 91})
        values = values.where(values.between(0, 180)).clip(upper=cap)
        # if 0 meals outside, fast food meals is 0 too
        if code == "DBD900":
            values = values.mask(meals.DBD895.round().eq(0), 0)
        note = "Son 7 gün. 22 = 22 veya daha fazla." if weekly else "Son 30 gün. 91 = 91 veya daha fazla."
        put(name, title, code, note, values)
    # home smoking question changed after 2011 (SMD410 -> SMD460 and SMD470)
    smoke_code = "SMD410" if year == 2011 else "SMD470"
    if year == 2011:
        smoke = read("SMQFAM", smoke_code)[smoke_code].round().map({1: 1, 2: 0})
    else:
        household = read("SMQFAM", "SMD460", smoke_code).round()
        smoke = household[smoke_code].map({0: 0, 1: 1, 2: 1, 3: 1}).mask(household.SMD460.eq(0), 0)
    put("smoking_inside_home", "Ev içinde tütün içilmesi", "SMD410 / SMD460 + SMD470", "1 = İçeride içen hane üyesi var, 0 = yok. 2013 sonrası diğer tütün türleri de dahil; evde içen hane üyesi olmayanlarda takip sorusu atlanır.", smoke)
    put("sweetened_beverages_day1_g", "Şekerli içecek, 1. gün (g)", "DR1IFF / WWEIA", "USDA sınıfları 7202, 7204, 7206. İlk geçerli beslenme görüşmesinde gazlı içecek, meyveli içecek ve spor/enerji içeceği gramı.", sweetened_drink_grams(folder, suffix, data.index))
    return pd.DataFrame(features, index=data.index).reset_index(drop=True), guide
