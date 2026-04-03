import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import math
import os

# ─────────────────────────────────────────────────────────────
# دالة مساعدة لتحويل الصورة إلى base64
# ─────────────────────────────────────────────────────────────
def image_to_base64(path: str) -> str:
    """تحوّل ملف صورة إلى سلسلة base64 لاستخدامها مباشرة في HTML."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

# ─────────────────────────────────────────────────────────────
# مسارات الشعارات  (ضع الملفات بجانب app.py)
# ─────────────────────────────────────────────────────────────
SEEMA_LOGO_PATH = os.path.join(os.path.dirname(__file__), "seema_logo.png")
GASTAT_LOGO_PATH = os.path.join(os.path.dirname(__file__), "gastat_logo.png")

SEEMA_LOGO_B64  = image_to_base64(SEEMA_LOGO_PATH)
GASTAT_LOGO_B64 = image_to_base64(GASTAT_LOGO_PATH)

# ── إحداثيات المدن ──
CITY_COORDS = {
    "الرياض":         (24.7136, 46.6753),
    "جدة":            (21.4858, 39.1925),
    "الدمام":         (26.4207, 50.0888),
    "مكة المكرمة":    (21.3891, 39.8579),
    "المدينة المنورة":(24.5247, 39.5692),
    "تبوك":           (28.3838, 36.5550),
    "أبها":           (18.2164, 42.5053),
    "بريدة":          (26.3292, 43.9749),
    "حائل":           (27.5114, 41.7208),
    "الخبر":          (26.2172, 50.1971),
}
CITY_AIRPORTS = {
    "الرياض":"RUH","جدة":"JED","الدمام":"DMM","مكة المكرمة":"JED",
    "المدينة المنورة":"MED","تبوك":"TUU","أبها":"AHB",
    "بريدة":"ELQ","حائل":"HAS","الخبر":"DMM",
}
CITY_DISTRICTS = {
    "الرياض":         ["العليا","النخيل","الملقا","الروضة","السليمانية"],
    "جدة":            ["الزهراء","الروضة","النزهة","الصفا","الحمراء"],
    "الدمام":         ["الشاطئ","العزيزية","الفيصلية","النور","الروضة"],
    "مكة المكرمة":    ["العزيزية","النسيم","الزاهر","العوالي","أجياد"],
    "المدينة المنورة":["العوالي","قباء","العزيزية","النخيل","الدفاع"],
    "تبوك":           ["المروج","الأمل","النزهة","الورود","الصناعية"],
    "أبها":           ["المنهل","النخيل","الحزم","الربوة","السلامة"],
    "بريدة":          ["المطار","العزيزية","الشفاء","النهضة","الروضة"],
    "حائل":           ["الصالحية","الغزالة","السلام","النزهة","الحمراء"],
    "الخبر":          ["العقربية","الثقبة","الراكة","الكورنيش","الصفا"],
}
REVIEW_TEXTS = ["السائق ممتاز","خدمة رائعة","في الوقت","سائق محترف","شكراً","وصل سليم","تأخر قليلاً"]

# ─────────────────────────────────────────────────────────────
# دوال توليد البيانات التجريبية
# ─────────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def generate_flight_data(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    rows = []
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 1, 1)
    suspicious_ids = suspicious_ids or set()
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        n_flights = rng.integers(2, 6) if is_susp else rng.integers(0, 3)
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        for _ in range(n_flights):
            arr_city = rng.choice([c for c in cities if c != home_city])
            dep_time = base_date + timedelta(days=int(rng.integers(0,80)), hours=int(rng.integers(5,22)))
            c1, c2   = CITY_COORDS[home_city], CITY_COORDS[arr_city]
            dist_km  = float(haversine_distance(np.array([c1[0]]), np.array([c1[1]]), np.array([c2[0]]), np.array([c2[1]]))[0])
            duration = int(dist_km / 800 * 60 + 45)
            arr_time = dep_time + timedelta(minutes=duration)
            rows.append({"account_id":acc_id,"flight_date":dep_time.strftime("%Y-%m-%d"),"departure_time":dep_time.strftime("%H:%M"),"arrival_time":arr_time.strftime("%H:%M"),"departure_city":home_city,"arrival_city":arr_city,"duration_min":duration,"distance_km":round(dist_km,1),"airport_code":CITY_AIRPORTS.get(home_city,"UNK")})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["account_id","flight_date","departure_time","arrival_time","departure_city","arrival_city","duration_min","distance_km","airport_code"])

def generate_review_data(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 1, 1, 8, 0)
    suspicious_ids = suspicious_ids or set()
    rows = []
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        base_lat, base_lon = CITY_COORDS[home_city]
        curr_time = base_date
        for i in range(int(rng.integers(8, 20))):
            if is_susp and i > 0 and rng.random() < 0.25:
                other  = rng.choice([c for c in cities if c != home_city])
                r_lat  = CITY_COORDS[other][0] + rng.uniform(-0.05, 0.05)
                r_lon  = CITY_COORDS[other][1] + rng.uniform(-0.05, 0.05)
                gap    = int(rng.integers(3, 12))
                dist   = rng.choice(CITY_DISTRICTS.get(other, ["غير محدد"]))
            else:
                r_lat  = base_lat + rng.uniform(-0.1, 0.1)
                r_lon  = base_lon + rng.uniform(-0.1, 0.1)
                gap    = int(rng.integers(15, 55))
                dist   = rng.choice(CITY_DISTRICTS.get(home_city, ["غير محدد"]))
            curr_time += timedelta(minutes=gap)
            rows.append({"account_id":acc_id,"review_time":curr_time.strftime("%Y-%m-%d %H:%M"),"district":dist,"rating":int(rng.integers(3,6)),"review_text":rng.choice(REVIEW_TEXTS),"lat":round(float(r_lat),5),"lon":round(float(r_lon),5)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def generate_gps_traces(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 3, 1, 8, 0)
    suspicious_ids = suspicious_ids or set()
    rows = []
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        curr_lat, curr_lon = CITY_COORDS[home_city]
        curr_time = base_date
        for i in range(int(rng.integers(20, 40))):
            if is_susp and i > 0 and rng.random() < 0.08:
                other   = rng.choice([c for c in cities if c != home_city])
                new_lat = CITY_COORDS[other][0] + rng.uniform(-0.02, 0.02)
                new_lon = CITY_COORDS[other][1] + rng.uniform(-0.02, 0.02)
                gap_sec = int(rng.integers(30, 120))
            else:
                new_lat = curr_lat + rng.uniform(-0.01, 0.01)
                new_lon = curr_lon + rng.uniform(-0.01, 0.01)
                gap_sec = int(rng.integers(120, 500))
            curr_time += timedelta(seconds=gap_sec)
            rows.append({"account_id":acc_id,"timestamp":curr_time.strftime("%Y-%m-%d %H:%M:%S"),"lat":round(float(new_lat),6),"lon":round(float(new_lon),6),"accuracy_m":int(rng.integers(3,15))})
            curr_lat, curr_lon = new_lat, new_lon
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# إعداد الصفحة
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="سِيماء — مرصد السلوك الرقمي",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CSS عام — RTL + هوية بصرية موحدة + طباعة
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── استيراد الخط ── */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

/* ── قاعدة RTL ── */
html, body, [class*="css"], .stApp {
    direction: rtl !important;
    font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
}

/* ── ترتيب العناصر من اليمين ── */
.stColumns, .element-container, .stMarkdown,
.stDataFrame, .stTable, .stMetric,
.stAlert, .stTabs, .stButton {
    direction: rtl !important;
    text-align: right !important;
}

/* ── إصلاح زر رفع الملف ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button,
button[kind="secondary"] {
    direction: ltr !important;
    unicode-bidi: normal !important;
}
[data-testid="stFileUploaderDropzone"] {
    direction: rtl !important;
}

/* ── الشريط الجانبي ── */
[data-testid="stSidebar"] {
    direction: rtl !important;
    text-align: right !important;
    background: linear-gradient(180deg, #0F2B4A 0%, #1A3F6B 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #E2EAF4 !important;
    font-family: 'Cairo', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1E9ED4 !important;
    color: white !important;
    border-radius: 8px !important;
    width: 100% !important;
}

/* ── بطاقة الهيدر ── */
.header-card {
    background: linear-gradient(135deg, #0F2B4A 0%, #1A3F6B 60%, #0D4A78 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-direction: row-reverse;
    gap: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.header-card .header-text {
    direction: rtl;
    flex: 1;
}
.header-card h1 {
    color: #FFFFFF !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}
.header-card p {
    color: #93C5FD !important;
    font-size: 0.95rem !important;
    margin: 8px 0 0 0 !important;
}
      .header-logos {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-direction: row;
}
.header-logos .seema-logo {
    height: 110px;
    object-fit: contain;
    filter: brightness(1.1);
}
.header-logos .gastat-logo {
    height: 72px;
    object-fit: contain;
    filter: brightness(1.1);
}
            
.header-logos .divider-logo {
    width: 1px;
    height: 56px;
    background: rgba(255,255,255,0.25);
}

/* ── بطاقات المقاييس ── */
[data-testid="metric-container"] {
    background: #F8FAFF !important;
    border: 1px solid #DBEAFE !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    text-align: right !important;
    direction: rtl !important;
    box-shadow: 0 2px 8px rgba(30,58,138,0.06) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(30,58,138,0.12) !important;
}
[data-testid="metric-container"] label {
    direction: rtl !important;
    text-align: right !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    direction: rtl !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
    color: #1E3A8A !important;
}

/* ── التبويبات ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    direction: rtl !important;
    border-bottom: 2px solid #DBEAFE !important;
}
.stTabs [data-baseweb="tab"] {
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1E3A8A !important;
    color: white !important;
}

/* ── توضيح الرسم البياني ── */
.chart-caption {
    background: #EFF6FF;
    border-right: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: -8px 0 20px 0;
    color: #1E40AF;
    font-size: 0.85rem;
    direction: rtl;
    text-align: right;
}

/* ── بطاقة التنبيه المخصصة ── */
.alert-normal {
    background: #F0FDF4;
    border: 1px solid #86EFAC;
    border-right: 4px solid #22C55E;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    direction: rtl;
    text-align: right;
    color: #166534;
    font-size: 0.9rem;
}
.alert-warning {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-right: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    direction: rtl;
    text-align: right;
    color: #92400E;
    font-size: 0.9rem;
}
.alert-danger {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-right: 4px solid #EF4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    direction: rtl;
    text-align: right;
    color: #991B1B;
    font-size: 0.9rem;
}
.alert-info {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-right: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    direction: rtl;
    text-align: right;
    color: #1E40AF;
    font-size: 0.9rem;
}

/* ── بطاقة المؤشر ── */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(30,58,138,0.08);
    border: 1px solid #DBEAFE;
    text-align: right;
    direction: rtl;
    margin-bottom: 16px;
}
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1E3A8A;
    line-height: 1.1;
}
.kpi-card .kpi-label {
    font-size: 0.875rem;
    color: #6B7280;
    margin-top: 4px;
}

/* ── الجداول ── */
.stDataFrame {
    direction: rtl !important;
}
.stDataFrame thead th {
    background-color: #1E3A8A !important;
    color: white !important;
    text-align: right !important;
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
}
.stDataFrame tbody td {
    text-align: right !important;
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
}

/* ── عناوين الأقسام ── */
h1, h2, h3, h4, h5, h6 {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', sans-serif !important;
    color: #1E3A8A !important;
}

/* ── الأزرار ── */
.stButton > button {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(30,58,138,0.2) !important;
}

/* ── الفاصل ── */
hr {
    border-color: #E0E9F5 !important;
    margin: 24px 0 !important;
}

/* ══════════════════════════════════════════
   طباعة نظيفة واحترافية
══════════════════════════════════════════ */
@media print {
    @page {
        size: A4 portrait;
        margin: 18mm 15mm 18mm 15mm;
    }

    /* إخفاء عناصر غير مطلوبة */
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    .stButton,
    .stFileUploader,
    .stCheckbox,
    .stDownloadButton,
    header, footer,
    .stTabs [data-baseweb="tab-list"],
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* إعادة ضبط الخلفيات */
    body, .stApp, .main {
        background: white !important;
        color: #111 !important;
    }

    /* هيدر الطباعة */
    .print-header {
        display: flex !important;
        align-items: center;
        justify-content: space-between;
        flex-direction: row-reverse;
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 14px;
        margin-bottom: 22px;
    }
    .print-header img {
        height: 64px;
        object-fit: contain;
    }
    .print-header .print-title {
        text-align: center;
        flex: 1;
    }
    .print-header .print-title h2 {
        font-size: 1.3rem;
        color: #1E3A8A !important;
        margin: 0;
    }
    .print-header .print-title p {
        font-size: 0.8rem;
        color: #555;
        margin: 4px 0 0;
    }

    /* فوتر الطباعة */
    .print-footer {
        display: block !important;
        position: fixed;
        bottom: 10mm;
        left: 0; right: 0;
        text-align: center;
        font-size: 0.75rem;
        color: #777;
        border-top: 1px solid #ddd;
        padding-top: 6px;
    }

    /* إظهار توضيحات الرسوم */
    .chart-caption {
        border: 1px solid #BFDBFE !important;
    }

    /* كسر الصفحات بشكل مناسب */
    .page-break-before { page-break-before: always; }
    .avoid-break { page-break-inside: avoid; }

    /* حجم النص */
    p, li, td, th { font-size: 10pt !important; }
    h3 { font-size: 12pt !important; }
}

/* هيدر الطباعة مخفي افتراضيًا على الشاشة */
.print-header, .print-footer { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# هيدر الصفحة الرئيسية مع الشعارين
# ─────────────────────────────────────────────────────────────
seema_img_tag  = f'<img src="{SEEMA_LOGO_B64}"  alt="شعار سيماء"  class="seema-logo"  />' if SEEMA_LOGO_B64  else ""
gastat_img_tag = f'<img src="{GASTAT_LOGO_B64}" alt="شعار الهيئة" class="gastat-logo" />' if GASTAT_LOGO_B64 else ""
divider_tag    = '<div class="divider-logo"></div>' if (SEEMA_LOGO_B64 and GASTAT_LOGO_B64) else ""

st.markdown(f"""
<div class="header-card">
    <div class="header-logos">
        {seema_img_tag}
        {divider_tag}
        {gastat_img_tag}
    </div>
    <div class="header-text">
        <h1>سِيماء — مرصد السلوك الرقمي</h1>
        <p>مرصد ذكي لكشف التستر الرقمي في اقتصاد المنصات عبر تحليل المصادر غير التقليدية</p>
    </div>
</div>
""", unsafe_allow_html=True)

# هيدر الطباعة (يظهر فقط عند الطباعة)
st.markdown(f"""
<div class="print-header">
    <div>
        {gastat_img_tag}
    </div>
    <div class="print-title">
        <h2>سِيماء — مرصد السلوك الرقمي</h2>
        <p>تقرير تحليل التستر الرقمي في اقتصاد المنصات</p>
    </div>
    <div>
        {seema_img_tag}
    </div>
</div>
<div class="print-footer">
    صدر بواسطة نظام سِيماء | الهيئة العامة للإحصاء | هاكاثون 2025
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# المعايير الرسمية
# ─────────────────────────────────────────────────────────────
BENCHMARKS = {
    "suspicious_ratio": 0.14,
    "max_daily_hours":  14,
    "max_continuous":   10,
    "avg_orders":       10,
    "max_devices":      1,
    "source": "GASTAT Q3 2024 + هيئة النقل سبتمبر 2024 + Uber Global Report 2023",
}

# ─────────────────────────────────────────────────────────────
# خرائط أسماء الأعمدة
# ─────────────────────────────────────────────────────────────
ACCOUNTS_COL_MAP = {
    "رقم الحساب (Account ID)":                              "account_id",
    "حالة الحساب (Status)":                                  "status",
    "الطلبات المقبولة/يوم (Accepted Orders)":                "accepted_orders",
    "الطلبات الملغاة/يوم (Cancelled Orders)":                "cancelled_orders",
    "ساعات العمل اليومية (Daily Hours)":                     "daily_hours",
    "ساعات العمل الأسبوعية (Weekly Hours)":                  "weekly_hours",
    "تسجيل الدخول/الخروج (Logins)":                          "logins",
    "ساعات العمل المتواصلة (Continuous Work)":               "continuous_work_hours",
    "سرعة قبول الطلب (Acceptance Speed - sec)":              "acceptance_speed_sec",
    "سرعة رفض الطلب (Rejection Speed - sec)":                "rejection_speed_sec",
    "نسبة القبول اليومية (Acceptance Rate)":                  "acceptance_rate",
    "نسبة الرفض اليومية (Rejection Rate)":                    "rejection_rate",
    "عدد الأجهزة المسجلة (Device Count)":                    "device_count",
    "المدينة الرئيسية (Main City)":                          "city",
    "عدد المدن/24 ساعة (City Count)":                        "city_count_per_day",
    "متوسط وقت الراحة بين الطلبات (Rest Interval - min)":    "rest_interval_min",
    "إجمالي المسافة المقطوعة/يوم (Daily Distance - km)":     "daily_distance_km",
}
ORDERS_COL_MAP = {
    "معرّف الطلب":        "order_id",
    "رقم الحساب":         "account_id",
    "تاريخ الطلب":        "order_date",
    "وقت البدء":          "start_time",
    "وقت الانتهاء":       "end_time",
    "مدة الطلب (دقيقة)":  "duration_min",
    "مدينة الطلب":        "order_city",
    "حي الاستلام":        "pickup_district",
    "خط عرض الاستلام":    "pickup_lat",
    "خط طول الاستلام":    "pickup_lon",
    "خط عرض التسليم":     "dropoff_lat",
    "خط طول التسليم":     "dropoff_lon",
    "المسافة (كم)":       "distance_km",
    "حالة الطلب":         "status",
}
FINANCIAL_COL_MAP = {
    "رقم الحساب (Account ID)":                   "account_id",
    "عدد مصادر الشحن المختلفة":                   "charge_sources",
    "عدد مرات السحب يومياً":                      "daily_withdrawals",
    "مجموع المبالغ المسحوبة يومياً (ريال)":        "daily_withdrawn_amount",
    "توقيتات السحب":                              "withdrawal_timing",
    "النمو الشهري في الدخل (%)":                  "monthly_income_growth",
    "عدد تحديثات الحساب البنكي":                  "bank_updates",
}
FLIGHT_COL_MAP = {
    "رقم الحساب (Account ID)":             "account_id",
    "تاريخ الرحلة (Flight Date)":           "flight_date",
    "وقت المغادرة (Departure Time)":        "departure_time",
    "وقت الوصول (Arrival Time)":            "arrival_time",
    "مدينة المغادرة (Departure City)":      "departure_city",
    "مدينة الوصول (Arrival City)":          "arrival_city",
    "مدة الرحلة (Duration - min)":          "duration_min",
    "المسافة (Distance - km)":              "distance_km",
    "رمز المطار (Airport Code)":            "airport_code",
}
REVIEW_COL_MAP = {
    "review_id":"review_id", "order_id":"order_id", "account_id":"account_id",
    "تاريخ التقييم (Review Date)":"review_date",
    "وقت التقييم (Review Time)":"review_time",
    "الحي (District)":"district",
    "خط العرض (Lat)":"lat", "خط الطول (Lon)":"lon",
    "التقييم (Rating)":"rating",
    "نص التقييم (Review Text)":"review_text",
}
GPS_COL_MAP = {
    "gps_id":"gps_id","order_id":"order_id","account_id":"account_id",
    "التاريخ (Date)":"date",
    "الوقت (Timestamp)":"timestamp",
    "خط العرض (Lat)":"lat","خط الطول (Lon)":"lon",
    "دقة الإشارة (Accuracy - m)":"accuracy_m",
}


# ─────────────────────────────────────────────────────────────
# دوال التحليل
# ─────────────────────────────────────────────────────────────
def normalize_score(series):
    if series.max() == series.min():
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - series.min()) / (series.max() - series.min()) * 100).round(2)

def compute_zscore(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def generate_reason(row):
    reasons = []
    if row.get("daily_hours", 0) > BENCHMARKS["max_daily_hours"]:
        reasons.append(f"ساعات عمل مرتفعة ({row['daily_hours']:.1f} ساعة/يوم)")
    if row.get("continuous_work_hours", 0) > BENCHMARKS["max_continuous"]:
        reasons.append(f"عمل متواصل {row['continuous_work_hours']:.1f} ساعة بلا توقف")
    if row.get("device_count", 1) > BENCHMARKS["max_devices"]:
        reasons.append(f"أجهزة متعددة ({row['device_count']:.0f})")
    if row.get("city_count_per_day", 1) > 1:
        reasons.append(f"نشاط في {row['city_count_per_day']:.0f} مدن/24 ساعة")
    if row.get("rest_interval_min", 30) < 5:
        reasons.append(f"وقت راحة منخفض جداً ({row['rest_interval_min']:.1f} دقيقة)")
    if row.get("accepted_orders", 0) > 30:
        reasons.append(f"طلبات مرتفعة ({row['accepted_orders']:.0f}/يوم)")
    if row.get("charge_sources", 1) > 3:
        reasons.append(f"مصادر شحن متعددة ({row['charge_sources']:.0f})")
    if row.get("withdrawal_timing", "") == "على مدار 24 ساعة (نشاط مستمر)":
        reasons.append("سحوبات على مدار 24 ساعة")
    if row.get("monthly_income_growth", 0) > 50:
        reasons.append(f"نمو مالي غير طبيعي ({row['monthly_income_growth']:.0f}%)")
    if row.get("bank_updates", 0) > 3:
        reasons.append(f"تحديثات بنكية متكررة ({row['bank_updates']:.0f})")
    if row.get("flight_conflict_score", 0) > 0:
        reasons.append("تعارض مع تذاكر الطيران")
    if row.get("review_conflict_score", 0) > 0:
        reasons.append("تنقل مستحيل في التقييمات")
    if row.get("gps_conflict_score", 0) > 0:
        reasons.append("قفزات GPS مستحيلة")
    return " — ".join(reasons) if reasons else "سلوك ضمن النطاق الطبيعي"


def build_internal_scores(df, contamination):
    df["city_lat"] = df["city"].map(lambda c: CITY_COORDS.get(c, (24.0, 45.0))[0])
    df["city_lon"] = df["city"].map(lambda c: CITY_COORDS.get(c, (24.0, 45.0))[1])
    rng = np.random.default_rng(42)
    df["lat"] = df["city_lat"] + rng.uniform(-0.08, 0.08, len(df))
    df["lon"] = df["city_lon"] + rng.uniform(-0.08, 0.08, len(df))
    for col in ["daily_hours","continuous_work_hours","accepted_orders","device_count","city_count_per_day"]:
        if col in df.columns:
            df[f"z_{col}"] = compute_zscore(df[col]).round(3)
    df["dist_from_city_center_km"] = haversine_distance(
        df["lat"].values, df["lon"].values,
        df["city_lat"].values, df["city_lon"].values
    ).round(2)
    df["geo_anomaly_score"] = compute_zscore(df["dist_from_city_center_km"]).abs().round(3)
    feature_cols = [c for c in [
        "daily_hours","weekly_hours","accepted_orders","cancelled_orders",
        "continuous_work_hours","device_count","city_count_per_day",
        "rest_interval_min","daily_distance_km","logins",
        "acceptance_rate","rejection_rate","geo_anomaly_score",
    ] if c in df.columns]
    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"]         = model.fit_predict(df[feature_cols])
    df["model_score_raw"] = -model.decision_function(df[feature_cols])
    df["model_score"]     = normalize_score(df["model_score_raw"])
    df["rule_score"] = 0
    if "daily_hours"           in df.columns: df["rule_score"] += (df["daily_hours"]           > 14).astype(int) * 20
    if "continuous_work_hours" in df.columns: df["rule_score"] += (df["continuous_work_hours"] > 10).astype(int) * 20
    if "device_count"          in df.columns: df["rule_score"] += (df["device_count"]          >  1).astype(int) * 20
    if "city_count_per_day"    in df.columns: df["rule_score"] += (df["city_count_per_day"]    >  1).astype(int) * 15
    if "accepted_orders"       in df.columns: df["rule_score"] += (df["accepted_orders"]       > 30).astype(int) * 15
    if "logins"                in df.columns: df["rule_score"] += (df["logins"]                > 15).astype(int) * 10
    z_available = [f"z_{c}" for c in ["daily_hours","continuous_work_hours","device_count"] if f"z_{c}" in df.columns]
    df["z_score_signal"] = normalize_score(df[z_available].abs().mean(axis=1)) if z_available else 50.0
    df["internal_suspicion_score"] = (
        0.50 * df["model_score"] +
        0.30 * df["rule_score"]  +
        0.20 * df["z_score_signal"]
    ).round(2)
    return df


def add_financial_scores(df, fin_df):
    fin = fin_df.copy()
    fin["fin_score"] = 0
    if "charge_sources"        in fin.columns: fin["fin_score"] += (fin["charge_sources"]        >  3).astype(int) * 25
    if "daily_withdrawals"     in fin.columns: fin["fin_score"] += (fin["daily_withdrawals"]     >  5).astype(int) * 20
    if "monthly_income_growth" in fin.columns: fin["fin_score"] += (fin["monthly_income_growth"] > 50).astype(int) * 25
    if "bank_updates"          in fin.columns: fin["fin_score"] += (fin["bank_updates"]          >  3).astype(int) * 15
    if "withdrawal_timing"     in fin.columns:
        fin["fin_score"] += (fin["withdrawal_timing"] == "على مدار 24 ساعة (نشاط مستمر)").astype(int) * 15
    df = df.merge(
        fin[["account_id","fin_score","charge_sources","daily_withdrawals",
             "monthly_income_growth","bank_updates","withdrawal_timing"]],
        on="account_id", how="left"
    )
    df["fin_score"] = df["fin_score"].fillna(0)
    return df


def detect_flight_conflicts_v2(accounts_df, orders_df, flight_df):
    if flight_df.empty or orders_df.empty:
        return pd.DataFrame()
    conflicts = []
    for acc_id in accounts_df["account_id"].unique():
        acc_flights = flight_df[flight_df["account_id"] == acc_id]
        acc_orders  = orders_df[(orders_df["account_id"] == acc_id) & (orders_df["status"] == "مكتمل")]
        if acc_flights.empty or acc_orders.empty:
            continue
        for _, flight in acc_flights.iterrows():
            flight_date = str(flight["flight_date"])[:10]
            dep_time    = str(flight["departure_time"])[:5]
            arr_time    = str(flight["arrival_time"])[:5]
            same_day_orders = acc_orders[acc_orders["order_date"].astype(str).str[:10] == flight_date]
            if same_day_orders.empty:
                continue
            for _, order in same_day_orders.iterrows():
                order_start = str(order["start_time"])[:5]
                if order_start >= dep_time:
                    conflicts.append({
                        "account_id":     acc_id,
                        "order_id":       order["order_id"],
                        "conflict_type":  "نشاط أثناء سفر مسجّل",
                        "conflict_score": 60,
                        "flight_date":    flight_date,
                        "flight_route":   f"{flight['departure_city']} → {flight['arrival_city']}",
                        "departure_time": dep_time,
                        "order_time":     order_start,
                        "order_city":     order["order_city"],
                        "evidence": (
                            f"طلب {order['order_id']} نُفِّذ في {order['order_city']} "
                            f"الساعة {order_start} بينما الرحلة أقلعت من "
                            f"{flight['departure_city']} الساعة {dep_time} نفس اليوم"
                        ),
                    })
    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def detect_review_conflicts_v2(orders_df, review_df):
    if review_df.empty:
        return pd.DataFrame()
    needed = ["account_id", "review_date", "review_time", "lat", "lon"]
    if not all(c in review_df.columns for c in needed):
        return pd.DataFrame()
    conflicts = []
    MIN_IMPOSSIBLE_SPEED = 300
    for acc_id, group in review_df.groupby("account_id"):
        group = group.copy()
        try:
            group["datetime"] = pd.to_datetime(
                group["review_date"].astype(str) + " " + group["review_time"].astype(str),
                errors="coerce"
            )
            group = group.dropna(subset=["datetime", "lat", "lon"])
            group = group.sort_values("datetime").reset_index(drop=True)
        except Exception:
            continue
        for i in range(1, len(group)):
            try:
                prev = group.iloc[i - 1]
                curr = group.iloc[i]
                gap_min = (curr["datetime"] - prev["datetime"]).total_seconds() / 60
                if gap_min <= 0 or gap_min > 120:
                    continue
                dist_km = haversine_scalar(
                    float(prev["lat"]), float(prev["lon"]),
                    float(curr["lat"]),  float(curr["lon"])
                )
                if dist_km < 50:
                    continue
                speed = dist_km / (gap_min / 60)
                if speed > MIN_IMPOSSIBLE_SPEED:
                    conflicts.append({
                        "account_id":    acc_id,
                        "order_id":      curr.get("order_id", "-"),
                        "review_id":     curr.get("review_id", "-"),
                        "conflict_type": "تنقل مستحيل بين تقييمين متتاليين",
                        "conflict_score": min(60, round(speed / MIN_IMPOSSIBLE_SPEED * 40 + 20, 1)),
                        "gap_min":       round(gap_min, 1),
                        "dist_km":       round(dist_km, 2),
                        "speed_kmh":     round(speed, 1),
                        "evidence": (
                            f"تقييمان متتاليان لـ {acc_id} بينهما {dist_km:.0f} كم "
                            f"في {gap_min:.0f} دقيقة فقط (سرعة {speed:.0f} كم/ساعة)"
                        ),
                    })
            except Exception:
                continue
    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def detect_gps_conflicts_v2(orders_df, gps_df):
    if gps_df.empty:
        return pd.DataFrame()
    needed = ["account_id", "lat", "lon", "timestamp"]
    if not all(c in gps_df.columns for c in needed):
        return pd.DataFrame()
    conflicts = []
    MIN_IMPOSSIBLE_SPEED = 400
    gps_work = gps_df.copy()
    gps_work["timestamp"] = pd.to_datetime(gps_work["timestamp"], errors="coerce")
    gps_work = gps_work.dropna(subset=["timestamp", "lat", "lon"])
    for acc_id, group in gps_work.groupby("account_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i in range(1, len(group)):
            try:
                prev = group.iloc[i - 1]
                curr = group.iloc[i]
                gap_sec = (curr["timestamp"] - prev["timestamp"]).total_seconds()
                if gap_sec <= 0 or gap_sec > 3600:
                    continue
                dist_km = haversine_scalar(
                    float(prev["lat"]), float(prev["lon"]),
                    float(curr["lat"]),  float(curr["lon"])
                )
                if dist_km < 100:
                    continue
                speed = dist_km / (gap_sec / 3600)
                if speed > MIN_IMPOSSIBLE_SPEED:
                    conflicts.append({
                        "account_id":     acc_id,
                        "order_id":       curr.get("order_id", "-"),
                        "conflict_type":  "قفزة جغرافية مستحيلة في GPS",
                        "conflict_score": min(60, round(speed / MIN_IMPOSSIBLE_SPEED * 40 + 20, 1)),
                        "gap_sec":        round(gap_sec, 0),
                        "dist_km":        round(dist_km, 2),
                        "speed_kmh":      round(speed, 1),
                        "evidence": (
                            f"نقطتا GPS لـ {acc_id} بينهما {dist_km:.0f} كم "
                            f"في {gap_sec:.0f} ثانية فقط (سرعة {speed:.0f} كم/ساعة)"
                        ),
                    })
            except Exception:
                continue
    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def compute_final_scores(df, flight_conflicts, review_conflicts, gps_conflicts):
    for col, conflicts in [
        ("flight_conflict_score", flight_conflicts),
        ("review_conflict_score", review_conflicts),
        ("gps_conflict_score",    gps_conflicts),
    ]:
        if not conflicts.empty and "conflict_score" in conflicts.columns:
            scores = conflicts.groupby("account_id")["conflict_score"].sum().reset_index()
            scores.columns = ["account_id", col]
            df = df.merge(scores, on="account_id", how="left")
        else:
            df[col] = 0
        df[col] = df[col].fillna(0)
    df["external_suspicion_score"] = (
        df["flight_conflict_score"] * 0.35 +
        df["review_conflict_score"] * 0.35 +
        df["gps_conflict_score"]    * 0.30
    ).clip(upper=100).round(2)
    df["suspicion_score"] = (
        0.45 * df["internal_suspicion_score"] +
        0.20 * normalize_score(df["fin_score"]) +
        0.35 * df["external_suspicion_score"]
    ).round(2)
    df["risk_level"] = pd.cut(
        df["suspicion_score"], bins=[-1, 30, 60, 100],
        labels=["منخفض", "متوسط", "مرتفع"]
    )
    df["final_flag"] = np.where(
        (df["anomaly"] == -1) | (df["suspicion_score"] >= 60),
        "مشبوه", "طبيعي"
    )
    df["reason"] = df.apply(generate_reason, axis=1)
    return df


# ─────────────────────────────────────────────────────────────
# دالة إنشاء تقرير PDF — عربي كامل + شعارات
# ─────────────────────────────────────────────────────────────
def create_pdf_report(df, total, suspicious, estimated, gap_pct):
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, Image, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import datetime
    import tempfile

    # ── تجهيز معالجة العربية ──
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        def ar(text):
            """تحوّل النص العربي ليظهر بشكل صحيح في PDF."""
            try:
                reshaped = arabic_reshaper.reshape(str(text))
                return get_display(reshaped)
            except Exception:
                return str(text)
    except ImportError:
        def ar(text):
            return str(text)

    # ── تسجيل خط عربي من نظام Windows ──
    ARABIC_FONT      = "Helvetica"
    ARABIC_FONT_BOLD = "Helvetica-Bold"
    arabic_font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/Cairo-Regular.ttf"),
    ]
    for font_path in arabic_font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("ArabicFont", font_path))
                ARABIC_FONT      = "ArabicFont"
                ARABIC_FONT_BOLD = "ArabicFont"
                break
            except Exception:
                continue

    # ── ألوان ──
    COLOR_PRIMARY = colors.HexColor("#1E3A8A")
    COLOR_LIGHT   = colors.HexColor("#EFF6FF")
    COLOR_BORDER  = colors.HexColor("#BFDBFE")
    COLOR_MUTED   = colors.HexColor("#6B7280")
    COLOR_ROW_ALT = colors.HexColor("#F8FAFC")

    # ── أنماط النص ──
    title_style = ParagraphStyle(
        "title", fontSize=16, alignment=TA_CENTER,
        spaceAfter=6, fontName=ARABIC_FONT_BOLD, textColor=COLOR_PRIMARY
    )
    sub_style = ParagraphStyle(
        "sub", fontSize=10, alignment=TA_CENTER,
        spaceAfter=4, fontName=ARABIC_FONT, textColor=COLOR_MUTED
    )
    heading_style = ParagraphStyle(
        "heading", fontSize=13, spaceAfter=6, spaceBefore=10,
        fontName=ARABIC_FONT_BOLD, textColor=COLOR_PRIMARY, alignment=TA_RIGHT
    )
    small_style = ParagraphStyle(
        "small", fontSize=8, alignment=TA_CENTER,
        fontName=ARABIC_FONT, textColor=COLOR_MUTED
    )

    # ── دالة تحميل الشعارات بأمان ──
    def load_logo_for_pdf(path, width_cm, height_cm):
        if not path or not os.path.exists(path):
            return None
        ext = path.rsplit(".", 1)[-1].lower()
        try:
            if ext == "svg":
                try:
                    import cairosvg
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    cairosvg.svg2png(url=path, write_to=tmp.name,
                                     output_width=int(width_cm * 40))
                    return Image(tmp.name, width=width_cm*cm, height=height_cm*cm)
                except Exception:
                    return None
            else:
                return Image(path, width=width_cm*cm, height=height_cm*cm)
        except Exception:
            return None

    # ── دالة بناء جدول موحد ──
    def make_table(data_rows, col_widths):
        processed = []
        for row in data_rows:
            processed.append([ar(c) if isinstance(c, str) else c for c in row])
        t = Table(processed, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  COLOR_PRIMARY),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  ARABIC_FONT_BOLD),
            ("FONTNAME",      (0,1), (-1,-1), ARABIC_FONT),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, COLOR_ROW_ALT]),
            ("GRID",          (0,0), (-1,-1), 0.4, COLOR_BORDER),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWHEIGHT",     (0,0), (-1,-1), 20),
        ]))
        return t

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ── هيدر الشعارات ──
    seema_img_elem  = load_logo_for_pdf(SEEMA_LOGO_PATH,  3.0, 2.0)
    gastat_img_elem = load_logo_for_pdf(GASTAT_LOGO_PATH, 3.5, 2.0)

    header_data = [[
        gastat_img_elem or "",
        [
            Paragraph(ar("سيماء — مرصد السلوك الرقمي"), title_style),
            Paragraph(ar("تقرير تحليل التستر الرقمي في اقتصاد المنصات"), sub_style),
            Paragraph(ar(f"تاريخ التقرير: {datetime.date.today().strftime('%Y-%m-%d')}"), sub_style),
        ],
        seema_img_elem or "",
    ]]
    header_table = Table(header_data, colWidths=[4*cm, 9*cm, 4*cm])
    header_table.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",         (0,0), (0,0),   "LEFT"),
        ("ALIGN",         (1,0), (1,0),   "CENTER"),
        ("ALIGN",         (2,0), (2,0),   "RIGHT"),
        ("BACKGROUND",    (0,0), (-1,-1), COLOR_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY, spaceAfter=14))

    # ── الملخص التنفيذي ──
    story.append(Paragraph(ar("الملخص التنفيذي"), heading_style))
    bench_pct    = BENCHMARKS["suspicious_ratio"] * 100
    gap_official = round(gap_pct - bench_pct, 2)
    gap_sign     = "+" if gap_official > 0 else ""
    story.append(make_table([
        ["المؤشر",                   "القيمة",                     "الدلالة"],
        ["اجمالي الحسابات المحللة",  str(total),                   "جميع الحسابات المدرجة في النظام"],
        ["الحسابات المشبوهة",        str(suspicious),              "تجاوزت حد الاشتباه 60 فاعلى"],
        ["العمالة المقدرة الفعلية",  str(estimated),               "بعد استبعاد الحسابات المشبوهة"],
        ["نسبة الفجوة المرصودة",     f"{gap_pct}%",                "نسبة الحسابات المشبوهة من الاجمالي"],
        ["المعدل الرسمي GASTAT",     f"{bench_pct}%",              "Q3 2024"],
        ["الفجوة عن المرجع الرسمي", f"{gap_sign}{gap_official}%", "موجب = اشتباه اعلى من المعدل الوطني"],
    ], [5.5*cm, 3.5*cm, 8*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── منهجية الكشف ──
    story.append(Paragraph(ar("منهجية الكشف - 3 طبقات"), heading_style))
    story.append(make_table([
        ["الطبقة",          "الوزن", "المكونات"],
        ["الطبقة الداخلية", "45%",   "Isolation Forest + قواعد رسمية + Z-Score + Haversine"],
        ["الطبقة المالية",  "20%",   "مصادر الشحن + توقيتات السحب + النمو الشهري + تحديثات البنك"],
        ["الطبقة الخارجية","35%",   "تذاكر الطيران + تقييمات التطبيقات + مسارات GPS"],
    ], [4*cm, 2.5*cm, 10.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── أعلى 10 حسابات مشبوهة ──
    story.append(Paragraph(ar("اعلى 10 حسابات مشبوهة"), heading_style))
    top_df = (df[df["final_flag"] == "مشبوه"]
              .sort_values("suspicion_score", ascending=False).head(10))
    top_rows = [["رقم الحساب", "المدينة", "الساعات/يوم", "الاجهزة",
                 "الداخلية",   "المالية", "الخارجية",    "النهائية"]]
    for _, row in top_df.iterrows():
        top_rows.append([
            str(row.get("account_id", "-")),
            str(row.get("city", "-")),
            f"{row.get('daily_hours', 0):.1f}",
            f"{row.get('device_count', 1):.0f}",
            f"{row.get('internal_suspicion_score', 0):.1f}",
            f"{row.get('fin_score', 0):.1f}",
            f"{row.get('external_suspicion_score', 0):.1f}",
            f"{row.get('suspicion_score', 0):.1f}",
        ])
    story.append(make_table(top_rows,
                            [2.5*cm, 2.5*cm, 2.2*cm, 2*cm,
                             2.2*cm, 2*cm,   2.2*cm, 2.2*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── التوصيات ──
    story.append(Paragraph(ar("التوصيات التشغيلية"), heading_style))
    top_city_risk = (
        df.groupby("city").agg(
            total=("account_id", "count"),
            suspicious=("final_flag", lambda x: (x == "مشبوه").sum())
        ).assign(risk_pct=lambda x: x["suspicious"] / x["total"] * 100)
        .sort_values("risk_pct", ascending=False)
    )
    top_city = top_city_risk.index[0] if len(top_city_risk) > 0 else "-"
    story.append(make_table([
        ["#", "التوصية"],
        ["1", f"مراجعة النشاط في مدينة {top_city} لارتفاع نسبة الاشتباه فيها"],
        ["2", "الحسابات ذات مصادر شحن اكثر من 3 او سحوبات على مدار 24 ساعة تستحق تدقيقا فوريا"],
        ["3", "ربط سجل الطلبات بتذاكر الطيران يكشف تعارضات لا ترصدها التحليلات السلوكية وحدها"],
        ["4", "عدم الاكتفاء بالعد الاداري — الحساب الواحد قد يمثل اكثر من عامل"],
    ], [1*cm, 16*cm]))

    # ── فوتر ──
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceAfter=8))
    story.append(Paragraph(
        ar(f"صدر بواسطة نظام سيماء | الهيئة العامة للاحصاء | هاكاثون 2025 | {datetime.date.today().strftime('%Y-%m-%d')}"),
        small_style
    ))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# الشريط الجانبي
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 12px 0 8px; direction:rtl;">
    <div style="font-size:1.2rem; font-weight:800; color:#93C5FD;">سِيماء</div>
    <div style="font-size:0.75rem; color:#93C5FD; opacity:0.8;">مرصد السلوك الرقمي</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📁 رفع البيانات")

uploaded_platform = st.sidebar.file_uploader(
    "📊 ملف المنصة — من Uber/Careem (Excel)",
    type=["xlsx"],
    help="seema_comprehensive_activity_data.xlsx — يحتوي 3 أوراق"
)
uploaded_flights = st.sidebar.file_uploader(
    "✈️ تذاكر الطيران — من GACA (Excel)", type=["xlsx"],
    help="flight_data.xlsx"
)
uploaded_reviews = st.sidebar.file_uploader(
    "📱 تقييمات التطبيقات — من Google Maps (Excel)", type=["xlsx"],
    help="review_data.xlsx"
)
uploaded_gps = st.sidebar.file_uploader(
    "🗺️ مسارات GPS — من Strava/OSM (Excel)", type=["xlsx"],
    help="gps_data.xlsx"
)

st.sidebar.markdown("---")
contamination = 0.10
show_only_suspicious = st.sidebar.checkbox("عرض الحسابات المشبوهة فقط", value=True)
st.sidebar.markdown("---")

n_uploaded = sum(x is not None for x in [uploaded_platform, uploaded_flights, uploaded_reviews, uploaded_gps])
if n_uploaded == 0:
    st.sidebar.warning("⬆️ ارفع ملف المنصة للبدء")
else:
    st.sidebar.success(f"✅ تم رفع {n_uploaded}/4 ملفات")

# معلومات المعايير في الشريط الجانبي
st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 المعايير الرسمية")
st.sidebar.markdown("""
<div style="direction:rtl; font-size:0.8rem; color:#CBD5E1; line-height:1.8;">
🕐 الحد الأقصى للعمل اليومي: <b style="color:#FCD34D;">14 ساعة</b><br>
⛔ العمل المتواصل: <b style="color:#FCD34D;">10 ساعات</b><br>
📦 متوسط الطلبات: <b style="color:#FCD34D;">10 طلب/يوم</b><br>
📊 نسبة المشبوهين: <b style="color:#FCD34D;">~14%</b>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# شاشة الترحيب — قبل رفع الملف
# ─────────────────────────────────────────────────────────────
if uploaded_platform is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:55vh;text-align:center;direction:rtl;">
        <div style="font-size:72px;margin-bottom:24px;">📂</div>
        <h2 style="color:#1E3A8A;margin-bottom:12px;font-family:'Cairo',sans-serif;">ارفع ملف المنصة للبدء</h2>
        <p style="color:#6B7280;font-size:1rem;max-width:500px;line-height:1.7;">
            ارفع ملف <b>seema_comprehensive_activity_data.xlsx</b> من القائمة الجانبية
            لبدء تحليل بيانات السائقين وكشف التستر الرقمي
        </p>
        <p style="color:#9CA3AF;font-size:0.875rem;margin-top:8px;">
            يمكنك إضافة ملفات الطيران والتقييمات والـ GPS لتحليل أعمق وأكثر دقة
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# قراءة البيانات
# ─────────────────────────────────────────────────────────────
raw_accounts = pd.read_excel(uploaded_platform, sheet_name="بيانات الحسابات (Accounts)")
accounts_df  = raw_accounts.rename(columns=ACCOUNTS_COL_MAP)

raw_orders = pd.read_excel(uploaded_platform, sheet_name="سجل الطلبات (Orders Log)")
orders_df  = raw_orders.rename(columns=ORDERS_COL_MAP)

raw_fin = pd.read_excel(uploaded_platform, sheet_name="البيانات المالية (Financial)")
fin_df  = raw_fin.rename(columns=FINANCIAL_COL_MAP)

st.sidebar.success(f"✅ تم رفع {n_uploaded}/4 ملفات")
st.sidebar.caption("المصدر: 📂 بيانات مرفوعة")

account_ids = accounts_df["account_id"].tolist()
city_map    = dict(zip(accounts_df["account_id"], accounts_df["city"]))

if uploaded_flights is not None:
    flight_df = pd.read_excel(uploaded_flights).rename(columns=FLIGHT_COL_MAP)
    flight_df["flight_date"] = flight_df["flight_date"].astype(str).str[:10]
else:
    flight_df = generate_flight_data(account_ids, city_map=city_map)
    flight_df["flight_date"] = flight_df["flight_date"].astype(str).str[:10] if "flight_date" in flight_df.columns else ""

if uploaded_reviews is not None:
    review_df = pd.read_excel(uploaded_reviews).rename(columns=REVIEW_COL_MAP)
else:
    review_df = generate_review_data(account_ids, city_map=city_map)

if uploaded_gps is not None:
    gps_df = pd.read_excel(uploaded_gps).rename(columns=GPS_COL_MAP)
    gps_df["timestamp"] = pd.to_datetime(
        gps_df["date"].astype(str) + " " + gps_df["timestamp"].astype(str), errors="coerce"
    )
else:
    gps_df = generate_gps_traces(account_ids, city_map=city_map)
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], errors="coerce")


# ─────────────────────────────────────────────────────────────
# التحليل
# ─────────────────────────────────────────────────────────────
with st.spinner("جاري التحليل، يرجى الانتظار..."):
    df = build_internal_scores(accounts_df.copy(), contamination)
    df = add_financial_scores(df, fin_df)
    flight_conflicts = detect_flight_conflicts_v2(df, orders_df, flight_df) if not orders_df.empty else pd.DataFrame()
    review_conflicts = detect_review_conflicts_v2(orders_df, review_df) if not orders_df.empty else pd.DataFrame()
    gps_conflicts    = detect_gps_conflicts_v2(orders_df, gps_df) if not orders_df.empty else pd.DataFrame()
    df = compute_final_scores(df, flight_conflicts, review_conflicts, gps_conflicts)

# ─────────────────────────────────────────────────────────────
# إحصاءات عامة
# ─────────────────────────────────────────────────────────────
total_accounts      = len(df)
suspicious_accounts = int((df["final_flag"] == "مشبوه").sum())
estimated_workers   = total_accounts - suspicious_accounts
gap_pct             = round(suspicious_accounts / total_accounts * 100, 2)
official_pct        = BENCHMARKS["suspicious_ratio"] * 100
gap_vs_official     = round(gap_pct - official_pct, 2)
display_df          = df[df["final_flag"] == "مشبوه"].copy() if show_only_suspicious else df.copy()


# ─────────────────────────────────────────────────────────────
# التبويبات
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 النظرة العامة",
    "🌐 المصادر غير التقليدية",
    "🔍 محرك الكشف",
    "📋 دعم القرار",
    "📄 التقارير",
])


# ══════════════════════════════════════════════════════════════
# Tab 1: النظرة العامة
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 النظرة العامة")

    # ── بطاقة الفجوة الإحصائية مع تفسير مبسّط ──
    if gap_vs_official > 5:
        alert_class = "alert-danger"
        interp_icon = "🔴"
        interp_text = "النسبة المرصودة أعلى من المعدل الوطني — هذا مؤشر على وجود تستر رقمي محتمل يستحق التدقيق"
    elif abs(gap_vs_official) <= 5:
        alert_class = "alert-normal"
        interp_icon = "🟢"
        interp_text = "النسبة المرصودة تتوافق مع المعدل الوطني — الأوضاع ضمن النطاق الطبيعي"
    else:
        alert_class = "alert-warning"
        interp_icon = "🟡"
        interp_text = "النسبة أقل من المعدل الوطني — البيانات أقل تشوهًا من المتوسط"

    st.markdown(f"""
    <div class="{alert_class}">
        <b>{interp_icon} الفجوة الإحصائية المرصودة: {gap_pct}%</b><br>
        المعدل الرسمي (GASTAT Q3 2024): {official_pct}% —
        الفجوة: <b>{gap_vs_official:+.2f}%</b><br>
        <span style="font-size:0.85rem;">{interp_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── توضيح للمستخدم العادي ──
    st.markdown("""
    <div class="alert-info">
        <b>💡 ماذا تعني هذه الأرقام؟</b><br>
        <span style="font-size:0.88rem;">
        سِيماء تحلل بيانات السائقين من عدة مصادر لاكتشاف الحسابات التي يبدو أنها تُستخدم من أكثر من شخص (التستر الرقمي).
        الحسابات "المشبوهة" تجاوزت مؤشرات تنبيه متعددة كساعات العمل الطويلة جداً، واستخدام أجهزة متعددة، والتعارض مع رحلات الطيران.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── مقاييس رئيسية ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 إجمالي الحسابات",    total_accounts)
    c2.metric("⚠️ الحسابات المشبوهة",  suspicious_accounts)
    c3.metric("👷 العمالة المقدّرة",    estimated_workers)
    c4.metric("📈 الفجوة المرصودة",     f"{gap_pct}%")
    c5.metric("📏 المعدل الرسمي",       f"{official_pct}%")

    st.write("")

    # ── رسوم بيانية + توضيح ──
    col_a, col_b = st.columns(2)
    with col_a:
        city_counts = df.groupby("city", as_index=False)["account_id"].count().rename(columns={"account_id":"عدد"})
        fig_city = px.bar(city_counts, x="city", y="عدد",
                          title="توزيع الحسابات حسب المدينة",
                          labels={"city":"المدينة"},
                          color="عدد",
                          color_continuous_scale="Blues")
        fig_city.update_layout(
            font_family="Cairo",
            title_font_size=14,
            xaxis_title="المدينة",
            yaxis_title="عدد الحسابات",
        )
        st.plotly_chart(fig_city, use_container_width=True)
        st.markdown("""
        <div class="chart-caption">
            📌 يوضح هذا الرسم عدد حسابات السائقين في كل مدينة. المدن ذات الأعمدة الأطول تحتوي على أكبر عدد من الحسابات وتستوجب رقابة أكبر.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        rs = df["final_flag"].value_counts().reset_index()
        rs.columns = ["flag", "count"]
        fig_pie = px.pie(rs, names="flag", values="count",
                         title="طبيعي مقابل مشبوه",
                         color="flag",
                         color_discrete_map={"طبيعي": "#22C55E", "مشبوه": "#EF4444"})
        fig_pie.update_layout(font_family="Cairo", title_font_size=14)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("""
        <div class="chart-caption">
            📌 يُظهر هذا الرسم نسبة الحسابات الطبيعية (خضراء) مقابل المشبوهة (حمراء). كلما زادت المساحة الحمراء، ارتفع احتمال وجود تستر رقمي.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 أهم 10 حسابات عالية الاشتباه")
    st.markdown("""
    <div class="alert-warning">
        ⚠️ هذه الحسابات سجّلت أعلى درجات اشتباه — يُنصح بمراجعتها أولاً. درجة الاشتباه من 0 إلى 100؛ كلما ارتفعت، زادت الحاجة للتدقيق.
    </div>
    """, unsafe_allow_html=True)
    top10_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","internal_suspicion_score",
        "external_suspicion_score","suspicion_score","risk_level","reason"
    ] if c in df.columns]
    st.dataframe(
        df.sort_values("suspicion_score", ascending=False)[top10_cols].head(10),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════
# Tab 2: المصادر غير التقليدية
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌐 المصادر غير التقليدية")
    st.markdown("""
    <div class="alert-info">
        💡 سِيماء تكشف التستر الرقمي عبر <b>4 مصادر بيانات</b>. كل مصدر يضيف دليلاً إضافياً على احتمالية استخدام الحساب من أكثر من شخص.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 المالية المشبوهة",    int((df["fin_score"] > 40).sum()))
    m2.metric("✈️ تعارضات الطيران",     len(flight_conflicts) if not flight_conflicts.empty else 0)
    m3.metric("📱 تعارضات التقييمات",   len(review_conflicts) if not review_conflicts.empty else 0)
    m4.metric("🗺️ شذوذات GPS",           len(gps_conflicts)    if not gps_conflicts.empty    else 0)

    # ── المصدر المالي ──
    st.markdown("---")
    st.markdown("### 💰 المصدر 1 — البيانات المالية")
    st.markdown("""
    <div class="alert-info">
        <b>ما يكشفه هذا المصدر:</b> إذا كان حساب السائق يُشحن من عدة محافظ مختلفة، أو تتم منه سحوبات في ساعات غير طبيعية (على مدار 24 ساعة)،
        أو نما دخله بشكل مفاجئ — فهذا مؤشر على أن أكثر من شخص يستخدم نفس الحساب.
    </div>
    """, unsafe_allow_html=True)
    fin_cols = [c for c in ["account_id","charge_sources","daily_withdrawals","monthly_income_growth","bank_updates","withdrawal_timing","fin_score"] if c in df.columns]
    fin_display = df[fin_cols].sort_values("fin_score", ascending=False)
    st.dataframe(fin_display.head(10), use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if "charge_sources" in df.columns:
            fig_cs = px.histogram(df, x="charge_sources", color="final_flag", nbins=15,
                                  title="توزيع مصادر الشحن",
                                  labels={"charge_sources":"عدد المصادر"},
                                  color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"})
            fig_cs.update_layout(font_family="Cairo")
            st.plotly_chart(fig_cs, use_container_width=True)
            st.markdown("""
            <div class="chart-caption">
                📌 الحسابات التي تمتلك أكثر من 3 مصادر شحن (يمين الرسم) تُعدّ مشبوهة لأن السائق الطبيعي لا يحتاج لعدة محافظ.
            </div>
            """, unsafe_allow_html=True)
    with col_f2:
        if "monthly_income_growth" in df.columns:
            fig_ig = px.histogram(df, x="monthly_income_growth", color="final_flag", nbins=20,
                                  title="توزيع النمو الشهري في الدخل",
                                  labels={"monthly_income_growth":"النمو (%)"},
                                  color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"})
            fig_ig.update_layout(font_family="Cairo")
            st.plotly_chart(fig_ig, use_container_width=True)
            st.markdown("""
            <div class="chart-caption">
                📌 نمو الدخل الشهري الطبيعي لا يتجاوز عادةً 50%. الحسابات التي تجاوزت هذا الحد (يمين الرسم) قد تشير لوجود أكثر من سائق.
            </div>
            """, unsafe_allow_html=True)

    # ── تذاكر الطيران ──
    st.markdown("---")
    st.markdown("### ✈️ المصدر 2 — تذاكر الطيران")
    st.markdown("""
    <div class="alert-info">
        <b>كيف يكشف التستر؟</b> إذا كان السائق في رحلة طيران مسجّلة، لكن حسابه نفّذ طلبات في نفس الوقت —
        فهذا دليل قاطع على أن شخصاً آخر يعمل على الحساب.
    </div>
    """, unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**بيانات الطيران الخام:**")
        st.dataframe(flight_df.head(8), use_container_width=True)
    with col_t2:
        if not flight_conflicts.empty:
            st.markdown("""
            <div class="alert-danger">
                🔴 <b>تعارضات مكتشفة!</b> هذه الحسابات نفّذت طلبات أثناء وجودها في رحلات طيران مسجّلة.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(flight_conflicts[["account_id","order_id","flight_route","departure_time","order_time","evidence"]].head(8), use_container_width=True)
        else:
            st.markdown("""
            <div class="alert-normal">
                🟢 لا توجد تعارضات مكتشفة حالياً — ارفع ملف المنصة مع ملف الطيران للكشف الحقيقي.
            </div>
            """, unsafe_allow_html=True)

    # ── تقييمات التطبيقات ──
    st.markdown("---")
    st.markdown("### 📱 المصدر 3 — تقييمات التطبيقات")
    st.markdown("""
    <div class="alert-info">
        <b>كيف يكشف التستر؟</b> إذا جاء تقييم العميل من موقع يستحيل الوصول إليه خلال الوقت المتاح بعد انتهاء الطلب —
        فهذا يعني أن شخصاً آخر في موقع مختلف أعطى التقييم.
    </div>
    """, unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**بيانات التقييمات الخام:**")
        st.dataframe(review_df.head(8), use_container_width=True)
    with col_r2:
        if not review_conflicts.empty:
            st.markdown("""
            <div class="alert-danger">
                🔴 <b>تنقلات مستحيلة!</b> هذه الحسابات أعطت تقييمات من مواقع لا يمكن الوصول إليها في الوقت المتاح.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(review_conflicts[["account_id","order_id","gap_min","dist_km","speed_kmh","evidence"]].head(8), use_container_width=True)
        else:
            st.markdown("""
            <div class="alert-normal">
                🟢 لا توجد تعارضات حالياً — ارفع ملف المنصة مع ملف التقييمات للكشف الحقيقي.
            </div>
            """, unsafe_allow_html=True)

    # ── GPS ──
    st.markdown("---")
    st.markdown("### 🗺️ المصدر 4 — مسارات GPS")
    st.markdown("""
    <div class="alert-info">
        <b>كيف يكشف التستر؟</b> إذا انتقلت إشارة GPS من موقع استلام الطلب إلى موقع آخر بعيد في ثوانٍ — فهذا مستحيل فيزيائياً،
        ويدل على أن الطلب بدأ من شخص مختلف.
    </div>
    """, unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**بيانات GPS الخام:**")
        st.dataframe(gps_df.head(8), use_container_width=True)
    with col_g2:
        if not gps_conflicts.empty:
            st.markdown("""
            <div class="alert-danger">
                🔴 <b>قفزات جغرافية مستحيلة!</b> هذه الحسابات سجّلت تنقلات تتجاوز حدود السرعة الممكنة.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(gps_conflicts[["account_id","order_id","gap_sec","dist_km","speed_kmh","evidence"]].head(8), use_container_width=True)
        else:
            st.markdown("""
            <div class="alert-normal">
                🟢 لا توجد شذوذات GPS حالياً — ارفع ملف المنصة مع ملف GPS للكشف الحقيقي.
            </div>
            """, unsafe_allow_html=True)

    # ── تأثير المصادر ──
    st.markdown("---")
    st.markdown("### ⚖️ تأثير المصادر الأربعة على درجة الاشتباه النهائية")
    fig_comp = px.scatter(
        df.sample(min(150, len(df))),
        x="internal_suspicion_score", y="suspicion_score",
        color="final_flag", size="fin_score",
        hover_data=["account_id","city","external_suspicion_score"],
        title="الدرجة الداخلية مقابل الدرجة النهائية",
        labels={"internal_suspicion_score":"الدرجة الداخلية","suspicion_score":"الدرجة النهائية"},
        color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"}
    )
    fig_comp.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                       line=dict(dash="dash", color="gray"))
    fig_comp.update_layout(font_family="Cairo")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
        📌 كل نقطة تمثل حساباً. النقاط الحمراء أعلى اليمين هي الأعلى اشتباهاً. حجم الدائرة يعكس درجة الاشتباه المالي.
        النقاط القريبة من الخط المتقطع تعني أن الدرجة الداخلية والنهائية متقاربتان.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Tab 3: محرك الكشف
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 محرك الكشف والتحليل")

    st.markdown("""
    <div class="alert-info">
        <b>🧠 كيف يعمل المحرك؟</b><br>
        <span style="font-size:0.88rem;">
        يعتمد سِيماء على ثلاث طبقات تحليلية مدمجة:
        الطبقة الداخلية (45%) تستخدم خوارزمية Isolation Forest للكشف عن الأنماط الشاذة، إلى جانب قواعد مبنية على المعايير الرسمية.
        الطبقة المالية (20%) تحلل مصادر الشحن وأنماط السحب.
        الطبقة الخارجية (35%) تربط البيانات بتذاكر الطيران، تقييمات التطبيقات، ومسارات GPS.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        fig_scatter = px.scatter(
            df, x="daily_hours", y="accepted_orders",
            color="final_flag",
            size="suspicion_score",
            hover_data=["account_id","city","device_count","reason"],
            title="ساعات العمل اليومية مقابل عدد الطلبات",
            color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"},
            labels={"daily_hours":"الساعات/يوم", "accepted_orders":"الطلبات/يوم"}
        )
        fig_scatter.update_layout(font_family="Cairo")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""
        <div class="chart-caption">
            📌 الحسابات الطبيعية (خضراء) تتركز في المنطقة الوسطى. الحسابات المشبوهة (حمراء وكبيرة الحجم) تتجاوز الحدود الرسمية
            لساعات العمل (14 ساعة/يوم) أو عدد الطلبات (10 طلبات/يوم).
        </div>
        """, unsafe_allow_html=True)

    with col_y:
        fig_hist = px.histogram(
            df, x="suspicion_score", color="final_flag",
            nbins=30, title="توزيع درجات الاشتباه النهائية",
            color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"},
            labels={"suspicion_score":"درجة الاشتباه"}
        )
        fig_hist.update_layout(font_family="Cairo")
        # خط الحد الفاصل
        fig_hist.add_vline(x=60, line_dash="dash", line_color="red",
                           annotation_text="حد الاشتباه (60)", annotation_position="top right")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("""
        <div class="chart-caption">
            📌 الحسابات التي تجاوزت الخط الأحمر (درجة 60 وما فوق) تُصنَّف مشبوهة. كلما ارتفع الشريط في المنطقة الحمراء،
            زاد عدد الحسابات ذات المخاطر العالية.
        </div>
        """, unsafe_allow_html=True)

    # ── Z-Score ──
    z_cols_available = [c for c in ["z_daily_hours","z_continuous_work_hours","z_device_count","z_accepted_orders"] if c in df.columns]
    if z_cols_available:
        st.markdown("---")
        st.markdown("### 📐 توزيع مؤشر الانحراف المعياري (Z-Score)")
        st.markdown("""
        <div class="alert-info">
            💡 مؤشر Z-Score يقيس مدى بُعد قيمة الحساب عن متوسط الحسابات الأخرى. الحسابات التي تتجاوز +2 أو -2 تُعدّ شاذة ومختلفة عن الطبيعي.
        </div>
        """, unsafe_allow_html=True)
        z_df = df[["final_flag"] + z_cols_available].melt(
            id_vars="final_flag", var_name="المؤشر", value_name="Z-Score"
        )
        fig_box = px.box(
            z_df, x="المؤشر", y="Z-Score", color="final_flag",
            title="توزيع Z-Score حسب تصنيف الحساب",
            color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"}
        )
        fig_box.update_layout(font_family="Cairo")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("""
        <div class="chart-caption">
            📌 الصناديق تُظهر توزيع قيم كل مؤشر. الصناديق الحمراء (المشبوهة) التي ترتفع بشكل كبير فوق الصناديق الخضراء
            تُشير إلى فجوة واضحة في السلوك بين الحسابات الطبيعية والمشبوهة.
        </div>
        """, unsafe_allow_html=True)

    # ── Haversine Map ──
    st.markdown("---")
    st.markdown("### 🗺️ التوزيع الجغرافي للحسابات")
    st.markdown("""
    <div class="alert-info">
        💡 الحسابات التي تنشط في مناطق بعيدة جداً عن مدينتها المسجّلة قد تعمل في مناطق مختلفة، مما يرفع احتمالية التستر.
    </div>
    """, unsafe_allow_html=True)
    fig_map = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="final_flag",
        size="geo_anomaly_score",
        hover_data=["account_id","city","dist_from_city_center_km"],
        zoom=5, center={"lat":24.5,"lon":44.0},
        mapbox_style="carto-positron",
        title="توزيع الحسابات جغرافياً",
        color_discrete_map={"طبيعي":"#22C55E","مشبوه":"#EF4444"}
    )
    fig_map.update_layout(font_family="Cairo")
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
        📌 كل نقطة على الخريطة تمثل حساباً. حجم النقطة يعكس مدى بُعده الجغرافي عن مدينته الأصلية. النقاط الحمراء الكبيرة تستحق المراجعة.
    </div>
    """, unsafe_allow_html=True)

    # ── Heatmap الارتباط ──
    st.markdown("---")
    st.markdown("### 🔥 خريطة الارتباط بين المؤشرات")
    st.markdown("""
    <div class="alert-info">
        💡 خريطة الارتباط تُظهر العلاقة بين المؤشرات المختلفة. الألوان الداكنة تعني ارتباطاً قوياً — أي أن المؤشرين يرتفعان أو ينخفضان معاً.
    </div>
    """, unsafe_allow_html=True)
    corr_cols = [c for c in [
        "daily_hours","accepted_orders","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","external_suspicion_score","suspicion_score"
    ] if c in df.columns]
    corr = df[corr_cols].corr()
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorscale="RdYlGn_r"
    ))
    fig_heat.update_layout(height=500, font_family="Cairo")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
        📌 القيم القريبة من +1 (أحمر داكن) تعني أن المؤشرين يرتفعان معاً — وهذا يساعد على فهم أي المؤشرات تتحرك في نفس الاتجاه.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 جدول النتائج الكاملة")
    result_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","flight_conflict_score",
        "review_conflict_score","gps_conflict_score",
        "internal_suspicion_score","external_suspicion_score",
        "suspicion_score","risk_level","final_flag","reason"
    ] if c in display_df.columns]
    st.dataframe(
        display_df[result_cols].sort_values("suspicion_score", ascending=False),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════
# Tab 4: دعم القرار
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 دعم القرار")

    st.markdown("""
    <div class="alert-info">
        💡 هذه الصفحة تُقدّم مقارنات وتوصيات عملية للمحقق أو صاحب القرار، استناداً إلى نتائج التحليل والمعايير الرسمية المعتمدة.
    </div>
    """, unsafe_allow_html=True)

    # ── مقارنة الأرقام ──
    st.markdown("### 📊 مقارنة الأرقام: الإداري vs سِيماء vs GASTAT")
    compare_df = pd.DataFrame({
        "الحالة": ["الرقم الإداري","بعد تنقية سِيماء","المعدل الرسمي (GASTAT)"],
        "القيمة": [
            total_accounts, estimated_workers,
            int(total_accounts * (1 - BENCHMARKS["suspicious_ratio"]))
        ]
    })
    fig_compare = px.bar(
        compare_df, x="الحالة", y="القيمة",
        text="القيمة", color="الحالة",
        title="مقارنة: الإداري مقابل سِيماء مقابل المعدل الرسمي",
        color_discrete_sequence=["#1E3A8A","#22C55E","#F59E0B"]
    )
    fig_compare.update_layout(font_family="Cairo", showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
        📌 الفجوة بين العمود الأول (الرقم الإداري) والعمود الثاني (بعد التنقية) تُمثّل الحسابات المشبوهة التي رصدها سِيماء.
        كلما زادت الفجوة، ارتفع احتمال وجود تستر رقمي.
    </div>
    """, unsafe_allow_html=True)

    # ── المعايير الرسمية ──
    st.markdown("---")
    st.markdown("### 📏 المعايير الرسمية المعتمدة")
    st.markdown("""
    <div class="alert-info">
        💡 هذه المعايير هي الحدود التي تعتمدها الجهات الرسمية للحكم على طبيعية نشاط السائق. أي حساب يتجاوزها يُعدّ مشبوهاً.
    </div>
    """, unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "المؤشر":  ["الحد الأقصى لساعات العمل اليومية","متوسط عدد الطلبات في اليوم","النسبة الوطنية للحسابات المشبوهة"],
        "القيمة":  ["14 ساعة/يوم","10 طلبات/يوم","~14%"],
        "المصدر":  ["هيئة النقل السعودية 2024","Uber Global Report 2023","GASTAT Q3 2024"],
    }))

    # ── المدن الأعلى خطورة ──
    st.markdown("---")
    st.markdown("### 🏙️ المدن الأعلى خطورة")
    city_risk = (
        df.groupby("city").agg(
            total=("account_id","count"),
            suspicious=("final_flag", lambda x: (x=="مشبوه").sum())
        ).reset_index()
    )
    city_risk["نسبة الاشتباه %"] = (city_risk["suspicious"] / city_risk["total"] * 100).round(2)
    city_risk = city_risk.sort_values("نسبة الاشتباه %", ascending=False)
    city_risk.columns = ["المدينة","إجمالي الحسابات","الحسابات المشبوهة","نسبة الاشتباه %"]
    st.dataframe(city_risk, use_container_width=True)

    top_city = city_risk.iloc[0]["المدينة"] if len(city_risk) > 0 else "—"
    top_pct  = city_risk.iloc[0]["نسبة الاشتباه %"] if len(city_risk) > 0 else 0

    # ── التوصيات ──
    st.markdown("---")
    st.markdown("### 💡 التوصيات التشغيلية")

    if top_pct > 20:
        city_alert_class = "alert-danger"
        city_alert_icon  = "🔴"
    elif top_pct > 14:
        city_alert_class = "alert-warning"
        city_alert_icon  = "🟡"
    else:
        city_alert_class = "alert-normal"
        city_alert_icon  = "🟢"

    st.markdown(f"""
    <div class="{city_alert_class}">
        {city_alert_icon} <b>توصية 1:</b> مراجعة النشاط في مدينة <b>{top_city}</b> — نسبة الاشتباه فيها {top_pct}%
        {'(أعلى من المعدل الوطني)' if top_pct > 14 else '(ضمن المعدل الوطني)'}.
    </div>
    <div class="alert-warning">
        🟡 <b>توصية 2:</b> الحسابات ذات مصادر شحن > 3 أو سحوبات على مدار 24 ساعة تستحق تدقيقاً فورياً.
    </div>
    <div class="alert-info">
        🔵 <b>توصية 3:</b> ربط سجل الطلبات بتذاكر الطيران كشف تعارضات مباشرة لا تُرصد بالتحليل السلوكي وحده.
    </div>
    <div class="alert-info">
        🔵 <b>توصية 4:</b> عدم الاكتفاء بالعدّ الإداري — الحساب الواحد قد يمثل أكثر من عامل.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Tab 5: التقارير
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📄 التقارير والمخرجات")

    # ── ملخص سريع ──
    st.markdown("### 📊 ملخص سريع")
    st.markdown("""
    <div class="alert-info">
        💡 هذا الملخص يجمع أهم المؤشرات في مكان واحد. يمكن طباعة هذه الصفحة مباشرة من المتصفح أو تحميل التقرير كـ PDF.
    </div>
    """, unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "المؤشر": [
            "إجمالي الحسابات","الحسابات المشبوهة","العمالة المقدّرة",
            "الفجوة المرصودة","المعدل الرسمي","الفجوة عن المرجع",
            "تعارضات الطيران","تعارضات التقييمات","شذوذات GPS",
        ],
        "القيمة": [
            total_accounts, suspicious_accounts, estimated_workers,
            f"{gap_pct}%", f"{official_pct}%", f"{gap_vs_official:+.2f}%",
            len(flight_conflicts) if not flight_conflicts.empty else 0,
            len(review_conflicts) if not review_conflicts.empty else 0,
            len(gps_conflicts)    if not gps_conflicts.empty    else 0,
        ],
        "الدلالة": [
            "إجمالي الحسابات المحللة",
            "تجاوزت درجة الاشتباه ≥ 60",
            "بعد استبعاد المشبوهة",
            "نسبة الاشتباه من الإجمالي",
            "المرجع الرسمي GASTAT Q3 2024",
            "موجب = أعلى من المعدل الوطني",
            "طلبات نُفّذت أثناء رحلات طيران",
            "تقييمات من مواقع مستحيلة",
            "قفزات جغرافية غير ممكنة",
        ]
    }))

    st.markdown("---")

    # ── تحميل CSV ──
    st.markdown("### 📥 تحميل النتائج")
    export_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","flight_conflict_score",
        "review_conflict_score","gps_conflict_score",
        "internal_suspicion_score","external_suspicion_score",
        "suspicion_score","risk_level","final_flag","reason"
    ] if c in df.columns]
    csv_data = df[export_cols].sort_values("suspicion_score", ascending=False).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 تحميل جميع النتائج (CSV)",
        data=csv_data,
        file_name="seema_results.csv",
        mime="text/csv",
        help="يمكن فتحه في Excel مباشرة"
    )

    st.markdown("---")

    # ── PDF ──
    st.markdown("### 🖨️ تقرير PDF رسمي")
    st.markdown("""
    <div class="alert-info">
        💡 التقرير يتضمن الشعارين (سِيماء وهيئة الإحصاء)، الملخص التنفيذي، منهجية الكشف، أعلى الحسابات المشبوهة، والتوصيات.
    </div>
    """, unsafe_allow_html=True)
    if st.button("📄 إنشاء تقرير PDF رسمي"):
        with st.spinner("جاري إنشاء التقرير..."):
            try:
                pdf_bytes = create_pdf_report(df, total_accounts, suspicious_accounts, estimated_workers, gap_pct)
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="seema_report.pdf" '
                    f'style="display:inline-block;background:#1E3A8A;color:white;padding:10px 24px;'
                    f'border-radius:8px;text-decoration:none;font-family:Cairo,sans-serif;font-weight:600;">'
                    f'📥 اضغط هنا لتحميل التقرير</a>',
                    unsafe_allow_html=True
                )
                st.success("✅ تم إنشاء التقرير بنجاح!")
            except Exception as e:
                st.error(f"حدث خطأ أثناء إنشاء التقرير: {e}")


